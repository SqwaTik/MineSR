package dev.minesr.client;

import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.buffers.Std140Builder;
import com.mojang.blaze3d.buffers.Std140SizeCalculator;
import com.mojang.blaze3d.pipeline.BindGroupLayout;
import com.mojang.blaze3d.pipeline.ColorTargetState;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.FilterMode;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import dev.minesr.Jitter;
import dev.minesr.NeuralWeights;
import dev.minesr.Quality;
import net.minecraft.client.renderer.MappableRingBuffer;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.resources.Identifier;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.joml.Vector4f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;

/**
 * Renders the world at a lower resolution and rebuilds the full-resolution image with a
 * temporal upscale. Everything here runs on the render thread.
 *
 * <p>Frame flow (driven by {@code GameRendererMixin}):
 * <ol>
 *   <li>{@link #jitterProjection}: picks this frame's sub-pixel offset, returns the jittered projection.</li>
 *   <li>{@link #beginLevel}: world rendering is redirected into the low-res target.</li>
 *   <li>{@link #endLevel}: temporal pass into the history buffer, then a sharpen pass into the
 *       game's full-res target. The hand and GUI are drawn after this, at full resolution.</li>
 * </ol>
 */
public final class Upscaler {
	private static final Logger LOG = LoggerFactory.getLogger("minesr");
	/** A camera jump bigger than this between frames is a teleport: start history over. */
	private static final float TELEPORT_DISTANCE = 32.0f;

	private static final ColorTargetState FLOAT_TARGET =
			new ColorTargetState(Optional.empty(), GpuFormat.RGBA16_FLOAT, ColorTargetState.WRITE_ALL);

	/** Per low-res pixel: where it was last frame. */
	private static final RenderPipeline MOTION = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/motion"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/motion"))
			.withBindGroupLayout(BindGroupLayout.builder()
					.withSampler("CurDepthSampler")
					.withUniform("UpscaleInfo", UniformType.UNIFORM_BUFFER)
					.build())
			.withColorTargetState(FLOAT_TARGET)
			.build();

	private static final RenderPipeline TEMPORAL = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/temporal"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/temporal"))
			.withBindGroupLayout(BindGroupLayout.builder()
					.withSampler("CurColorSampler")
					.withSampler("MotionSampler")
					.withSampler("HistorySampler")
					.withUniform("UpscaleInfo", UniformType.UNIFORM_BUFFER)
					.build())
			.withColorTargetState(FLOAT_TARGET)
			.build();

	/** Plain copy, for frame generation with upscaling off. */
	private static final RenderPipeline BLIT = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/passthrough"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/blit"))
			.withBindGroupLayout(BindGroupLayout.builder().withSampler("InSampler").build())
			.withColorTargetState(ColorTargetState.DEFAULT)
			.build();

	/** Neural step 1: the network, once per low-res pixel (decide.fsh), compiled for its size. */
	private static RenderPipeline decidePipeline(int hidden) {
		return RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
				.withLocation(id("pipeline/decide_" + hidden))
				.withShaderDefine("HIDDEN", hidden)
				.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
				.withFragmentShader(id("post/decide"))
				.withBindGroupLayout(BindGroupLayout.builder()
						.withSampler("CurColorSampler")
						.withSampler("MotionSampler")
						.withSampler("HistorySampler")
						.withUniform("UpscaleInfo", UniformType.UNIFORM_BUFFER)
						.withUniform("NetWeights", UniformType.UNIFORM_BUFFER)
						.build())
				.withColorTargetState(FLOAT_TARGET)
				.build();
	}

	/** Neural step 2: the temporal pass applying those decisions at full resolution (temporal_nn.fsh). */
	private static final RenderPipeline TEMPORAL_NN = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/temporal_nn"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/temporal_nn"))
			.withBindGroupLayout(BindGroupLayout.builder()
					.withSampler("CurColorSampler")
					.withSampler("MotionSampler")
					.withSampler("HistorySampler")
					.withSampler("DecisionSampler")
					.withUniform("UpscaleInfo", UniformType.UNIFORM_BUFFER)
					.build())
			.withColorTargetState(FLOAT_TARGET)
			.build();

	/** The first version (reprojection per output pixel, 9-tap history). Kept to benchmark against. */
	private static final RenderPipeline TEMPORAL_V1 = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/temporal_v1"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/temporal_v1"))
			.withBindGroupLayout(BindGroupLayout.builder()
					.withSampler("CurColorSampler")
					.withSampler("CurDepthSampler")
					.withSampler("HistorySampler")
					.withUniform("UpscaleInfo", UniformType.UNIFORM_BUFFER)
					.build())
			.withColorTargetState(FLOAT_TARGET)
			.build();

	private static final RenderPipeline SHARPEN = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/sharpen"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/sharpen"))
			.withBindGroupLayout(BindGroupLayout.builder()
					.withSampler("InSampler")
					.withUniform("SharpenInfo", UniformType.UNIFORM_BUFFER)
					.build())
			.withColorTargetState(ColorTargetState.DEFAULT)
			.build();

	private static final int UPSCALE_UBO_SIZE = new Std140SizeCalculator()
			.putMat4f().putMat4f().putVec4().putVec4().putVec4().putVec4().get();
	private static final int SHARPEN_UBO_SIZE = new Std140SizeCalculator().putVec4().get();

	private final MineSRConfig config;
	private final GpuTimer timer = new GpuTimer();
	private final Capture capture;
	private boolean captureFrame;

	private TextureTarget low;
	private GpuTexture motion;
	private GpuTextureView motionView;
	private GpuTexture decisions;          // neural: per low-res pixel sharpness, alpha, beta, memory
	private GpuTextureView decisionsView;
	private final GpuTexture[] history = new GpuTexture[2];
	private final GpuTextureView[] historyView = new GpuTextureView[2];
	private int outWidth, outHeight;
	private int current;
	private MappableRingBuffer upscaleUbo;
	private NeuralWeights net;        // null: no trained network available, hand-tuned pass only
	private RenderPipeline neuralPipeline;
	private String netSource = "none";
	private GpuBuffer netBuffer;
	private MappableRingBuffer sharpenUbo;

	private boolean pipelinesChecked;
	private String failure;

	// Per-frame state
	private boolean frameActive;
	private boolean levelActive;
	private long frame;
	private int phases = 8;
	private float jitterX, jitterY;
	private final Matrix4f viewProj = new Matrix4f();
	private final Matrix4f prevViewProj = new Matrix4f();
	private Vec3 cameraPos = Vec3.ZERO;
	private Vec3 prevCameraPos = Vec3.ZERO;
	/** False until a frame has been seen since allocation or reset; then "previous" values are real. */
	private boolean hasLast;
	private boolean historyValid;

	public Upscaler(MineSRConfig config) {
		this.config = config;
		this.capture = new Capture(config);
	}

	public Capture capture() {
		return capture;
	}

	private boolean passthrough;    // frame generation without upscaling: full resolution, no jitter
	private boolean motionReady;    // this frame's motion texture is valid
	private boolean continuous;     // this frame continues the previous one (no cut, teleport or reset)

	/** For frame generation: motion of this frame (per low-res pixel: previous uv - current uv). */
	public GpuTextureView motionView() {
		return motionView;
	}

	public boolean motionReadyThisFrame() {
		return motionReady && frameActiveForMotion;
	}

	public boolean continuousWithPreviousFrame() {
		return continuous;
	}

	public void setNetwork(NeuralWeights network, String source) {
		net = network;
		neuralPipeline = network == null ? null : decidePipeline(network.hidden());
		netSource = network == null ? source : source + ", " + network.hidden() + " hidden units";
		pipelinesChecked = false;
		if (netBuffer != null) {
			netBuffer.close();
			netBuffer = null;
		}
	}

	/** True when this frame will use the trained network. */
	public boolean neuralActive() {
		return config.neural && net != null && config.passVersion == 2;
	}

	public String networkSource() {
		return netSource;
	}

	private static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath("minesr", path);
	}

	public String failure() {
		return failure;
	}

	/** Texture derivative scale for the world shaders (see ShaderPatch); 1 when not upscaling. */
	public float lodScale() {
		if (capture.isActive()) {
			return capture.lodScale();
		}
		return config.enabled && config.lodBias && failure == null && !passthrough ? config.qualityPreset().scale : 1.0f;
	}

	public boolean isLevelActive() {
		return levelActive;
	}

	public RenderTarget lowTarget() {
		return captureFrame ? capture.target() : low;
	}

	public String describe() {
		if (failure != null) {
			return "disabled: " + failure;
		}
		if (!config.enabled) {
			return "off" + gpuSummary();
		}
		if (low == null) {
			return "waiting for the first frame in a world";
		}
		return String.format("%s (%s): %dx%d -> %dx%d, %d jitter phases, sharpness %.2f",
				config.qualityPreset().id(), neuralActive() ? "neural" : "hand-tuned",
				low.width, low.height, outWidth, outHeight, phases, config.sharpness) + gpuSummary();
	}

	/**
	 * Called with the projection the world is about to be drawn with. Allocates targets for the
	 * current window size, remembers the un-jittered matrices for reprojection, and returns the
	 * projection shifted by this frame's sub-pixel offset.
	 */
	public Matrix4f jitterProjection(Matrix4f projection, Matrix4fc viewRotation, Vec3 camera, int width, int height) {
		frameActive = false;
		captureFrame = false;
		if (capture.isActive() && failure == null && checkPipelines()) {
			capture.beginFrame(projection, viewRotation, camera);
			captureFrame = true;
			return projection; // jitter is applied offline, when the low-res inputs are cut out
		}
		// Frame generation needs motion even with upscaling off: then the world renders at full
		// resolution without jitter and passes straight through (the image stays vanilla).
		passthrough = !config.enabled && config.frameGeneration;
		if ((!config.enabled && !passthrough) || failure != null || !checkPipelines()) {
			return projection;
		}
		ensureTargets(width, height);

		Matrix4f current = new Matrix4f(projection).mul(viewRotation);
		prevViewProj.set(hasLast ? viewProj : current);
		viewProj.set(current);
		prevCameraPos = hasLast ? cameraPos : camera;
		cameraPos = camera;
		historyValid = historyValid && hasLast && camera.distanceTo(prevCameraPos) < TELEPORT_DISTANCE;
		hasLast = true;

		frame++;
		phases = Jitter.phaseCount(low.width, outWidth);
		float[] j = passthrough ? new float[2] : config.gridJitter
				? Jitter.gridOffset(frame, Jitter.gridFactor(low.width, outWidth))
				: Jitter.offset(frame, phases);
		jitterX = j[0];
		jitterY = j[1];
		frameActive = true;
		// Shift by jitter pixels of the low-res target: one pixel is 2/width in clip space.
		return new Matrix4f().translation(2.0f * jitterX / low.width, 2.0f * jitterY / low.height, 0.0f).mul(projection);
	}

	private boolean frameActiveForMotion;

	public void beginLevel() {
		frames++;
		frameActiveForMotion = frameActive;
		if (!frameActive) {
			motionReady = false;
		}
		timer.stamp(GpuTimer.WORLD_START);
		levelActive = frameActive || captureFrame;
	}

	public void endLevel(RenderTarget main) {
		timer.stamp(GpuTimer.WORLD_END);
		if (!levelActive) {
			return;
		}
		if (captureFrame) {
			levelActive = false;
			captureFrame = false;
			try {
				capture.endFrame(main);
			} catch (RuntimeException e) {
				LOG.error("MineSR capture failed, stopping it", e);
				capture.stop();
			}
			return;
		}
		levelActive = false;
		frameActive = false;
		try {
			runPasses(main);
			timer.stamp(GpuTimer.UPSCALE_END);
		} catch (RuntimeException e) {
			fail("upscale pass failed: " + e, e);
		}
	}

	private long frames;

	/** World frames rendered so far; for measuring frame rate over exact time windows. */
	public long frameCount() {
		return frames;
	}

	public void startGpuRecording() {
		timer.startRecording();
	}

	public java.util.List<double[]> stopGpuRecording() {
		return timer.stopRecording();
	}

	/** Smoothed GPU milliseconds for drawing the world (at whatever resolution it's drawn). */
	public double gpuWorldMs() {
		return timer.worldMs();
	}

	/** Smoothed GPU milliseconds for the upscale and sharpen passes; NaN while off. */
	public double gpuUpscaleMs() {
		return timer.upscaleMs();
	}

	private String gpuSummary() {
		if (Double.isNaN(timer.worldMs())) {
			return "";
		}
		return Double.isNaN(timer.upscaleMs())
				? String.format(", GPU: world %.2f ms", timer.worldMs())
				: String.format(", GPU: world %.2f ms + upscale %.2f ms (%s), CPU: pass %.2f ms",
						timer.worldMs(), timer.upscaleMs(), timer.stages(), cpuPassMs);
	}

	// CPU-side diagnostics (milliseconds, smoothed): total time in runPasses and the part spent
	// waiting for a uniform buffer the GPU is still reading.
	private double cpuPassMs = Double.NaN, cpuWaitMs = Double.NaN;

	private static double smooth(double old, double sample) {
		return Double.isNaN(old) ? sample : old + (sample - old) * 0.05;
	}

	private void runPasses(RenderTarget main) {
		long start = System.nanoTime();
		motionReady = false;
		CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
		int previous = current;
		current ^= 1;

		Matrix4f invViewProj = new Matrix4f(viewProj).invert();
		Vec3 delta = cameraPos.subtract(prevCameraPos);
		boolean zeroToOne = RenderSystem.getDevice().getDeviceInfo().isZZeroToOne();
		long waitStart = System.nanoTime();
		upscaleUbo.currentBuffer();
		sharpenUbo.currentBuffer();
		cpuWaitMs = smooth(cpuWaitMs, (System.nanoTime() - waitStart) / 1e6);
		try (GpuBufferSlice.MappedView view = upscaleUbo.currentBuffer().map(false, true)) {
			Std140Builder.intoBuffer(view.data())
					.putMat4f(invViewProj)
					.putMat4f(prevViewProj)
					.putVec4((float) delta.x, (float) delta.y, (float) delta.z, zeroToOne ? 1.0f : 0.0f)
					.putVec4(jitterX, jitterY, 0.0f, 0.0f)
					.putVec4(low.width, low.height, outWidth, outHeight)
					.putVec4(historyValid ? 1.0f : 0.0f, config.maxHistory, 0.0f, 0.0f);
		}
		try (GpuBufferSlice.MappedView view = sharpenUbo.currentBuffer().map(false, true)) {
			Std140Builder.intoBuffer(view.data()).putVec4(config.sharpness, 0.0f, 0.0f, 0.0f);
		}

		var samplers = RenderSystem.getSamplerCache();
		var nearest = samplers.getClampToEdge(FilterMode.NEAREST);
		boolean v1 = config.passVersion == 1;
		if (!v1) {
			try (RenderPass pass = encoder.createRenderPass(() -> "MineSR motion", motionView, Optional.empty())) {
				pass.setPipeline(MOTION);
				RenderSystem.bindDefaultUniforms(pass);
				pass.setUniform("UpscaleInfo", upscaleUbo.currentBuffer());
				pass.bindTexture("CurDepthSampler", low.getDepthTextureView(), nearest);
				pass.draw(3, 1, 0, 0);
			}
			timer.stamp(GpuTimer.MOTION_END);
			motionReady = true;
			continuous = historyValid;
		}
		if (passthrough) {
			try (RenderPass pass = encoder.createRenderPass(() -> "MineSR passthrough", main.getColorTextureView(), Optional.empty())) {
				pass.setPipeline(BLIT);
				RenderSystem.bindDefaultUniforms(pass);
				pass.bindTexture("InSampler", low.getColorTextureView(), nearest);
				pass.draw(3, 1, 0, 0);
			}
			upscaleUbo.rotate();
			sharpenUbo.rotate();
			historyValid = true;
			timer.stamp(GpuTimer.UPSCALE_END);
			return;
		}
		boolean neural = !v1 && neuralActive();
		if (neural && netBuffer == null) {
			ByteBuffer data = ByteBuffer.allocateDirect(net.data().length * 4).order(ByteOrder.nativeOrder());
			data.asFloatBuffer().put(net.data());
			netBuffer = RenderSystem.getDevice().createBuffer(() -> "MineSR NetWeights", GpuBuffer.USAGE_UNIFORM, data);
		}
		if (neural) {
			try (RenderPass pass = encoder.createRenderPass(() -> "MineSR decide", decisionsView, Optional.empty())) {
				pass.setPipeline(neuralPipeline);
				RenderSystem.bindDefaultUniforms(pass);
				pass.setUniform("UpscaleInfo", upscaleUbo.currentBuffer());
				pass.setUniform("NetWeights", netBuffer);
				pass.bindTexture("CurColorSampler", low.getColorTextureView(), nearest);
				pass.bindTexture("MotionSampler", motionView, nearest);
				pass.bindTexture("HistorySampler", historyView[previous], samplers.getClampToEdge(FilterMode.LINEAR));
				pass.draw(3, 1, 0, 0);
			}
			timer.stamp(GpuTimer.DECIDE_END);
		}
		try (RenderPass pass = encoder.createRenderPass(() -> "MineSR temporal", historyView[current], Optional.empty())) {
			pass.setPipeline(v1 ? TEMPORAL_V1 : neural ? TEMPORAL_NN : TEMPORAL);
			RenderSystem.bindDefaultUniforms(pass);
			pass.setUniform("UpscaleInfo", upscaleUbo.currentBuffer());
			if (neural) {
				pass.bindTexture("DecisionSampler", decisionsView, nearest);
			}
			pass.bindTexture("CurColorSampler", low.getColorTextureView(), nearest);
			if (v1) {
				pass.bindTexture("CurDepthSampler", low.getDepthTextureView(), nearest);
			} else {
				pass.bindTexture("MotionSampler", motionView, nearest);
			}
			pass.bindTexture("HistorySampler", historyView[previous], samplers.getClampToEdge(FilterMode.LINEAR));
			pass.draw(3, 1, 0, 0);
		}
		timer.stamp(GpuTimer.TEMPORAL_END);
		try (RenderPass pass = encoder.createRenderPass(() -> "MineSR sharpen", main.getColorTextureView(), Optional.empty())) {
			pass.setPipeline(SHARPEN);
			RenderSystem.bindDefaultUniforms(pass);
			pass.setUniform("SharpenInfo", sharpenUbo.currentBuffer());
			pass.bindTexture("InSampler", historyView[current], samplers.getClampToEdge(FilterMode.NEAREST));
			pass.draw(3, 1, 0, 0);
		}
		upscaleUbo.rotate();
		sharpenUbo.rotate();
		historyValid = true;
		cpuPassMs = smooth(cpuPassMs, (System.nanoTime() - start) / 1e6);
	}

	private void ensureTargets(int width, int height) {
		float scale = passthrough ? 1.0f : config.qualityPreset().scale;
		int lowWidth = Quality.scaled(width, scale);
		int lowHeight = Quality.scaled(height, scale);
		if (low != null && low.width == lowWidth && low.height == lowHeight && outWidth == width && outHeight == height) {
			return;
		}
		release();
		GpuDevice device = RenderSystem.getDevice();
		low = new TextureTarget("MineSR low-res world", lowWidth, lowHeight, true, GpuFormat.RGBA8_UNORM);
		int usage = GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_RENDER_ATTACHMENT | GpuTexture.USAGE_COPY_DST;
		motion = device.createTexture(() -> "MineSR motion", usage, GpuFormat.RGBA16_FLOAT, lowWidth, lowHeight, 1, 1);
		motionView = device.createTextureView(motion);
		decisions = device.createTexture(() -> "MineSR decisions", usage, GpuFormat.RGBA16_FLOAT, lowWidth, lowHeight, 1, 1);
		decisionsView = device.createTextureView(decisions);
		for (int i = 0; i < 2; i++) {
			int index = i;
			history[i] = device.createTexture(() -> "MineSR history " + index, usage, GpuFormat.RGBA16_FLOAT, width, height, 1, 1);
			historyView[i] = device.createTextureView(history[i]);
			device.createCommandEncoder().clearColorTexture(history[i], new Vector4f(0.0f));
		}
		upscaleUbo = new MappableRingBuffer(() -> "MineSR UpscaleInfo", GpuBuffer.USAGE_UNIFORM | GpuBuffer.USAGE_MAP_WRITE, UPSCALE_UBO_SIZE);
		sharpenUbo = new MappableRingBuffer(() -> "MineSR SharpenInfo", GpuBuffer.USAGE_UNIFORM | GpuBuffer.USAGE_MAP_WRITE, SHARPEN_UBO_SIZE);
		outWidth = width;
		outHeight = height;
		historyValid = false;
		LOG.info("MineSR: rendering world at {}x{}, output {}x{}", lowWidth, lowHeight, width, height);
	}

	/** Frees GPU resources. Safe to call repeatedly; the next frame reallocates if still enabled. */
	public void release() {
		if (low != null) {
			low.destroyBuffers();
			low = null;
		}
		if (motionView != null) {
			motionView.close();
			motionView = null;
		}
		if (motion != null) {
			motion.close();
			motion = null;
		}
		if (decisionsView != null) {
			decisionsView.close();
			decisionsView = null;
		}
		if (decisions != null) {
			decisions.close();
			decisions = null;
		}
		for (int i = 0; i < 2; i++) {
			if (historyView[i] != null) {
				historyView[i].close();
				historyView[i] = null;
			}
			if (history[i] != null) {
				history[i].close();
				history[i] = null;
			}
		}
		if (upscaleUbo != null) {
			upscaleUbo.close();
			upscaleUbo = null;
		}
		if (sharpenUbo != null) {
			sharpenUbo.close();
			sharpenUbo = null;
		}
		historyValid = false;
		hasLast = false;
	}

	/** Start accumulation over, e.g. after a dimension change or settings change. */
	public void resetHistory() {
		historyValid = false;
		hasLast = false;
		timer.reset();
	}

	/** Clears a previous shader failure so the next frame tries again (after a resource reload). */
	public void retry() {
		failure = null;
		pipelinesChecked = false;
	}

	private boolean checkPipelines() {
		if (pipelinesChecked) {
			return true;
		}
		GpuDevice device = RenderSystem.getDevice();
		if (!device.precompilePipeline(MOTION).isValid() || !device.precompilePipeline(TEMPORAL).isValid()
				|| !device.precompilePipeline(TEMPORAL_V1).isValid()
				|| !device.precompilePipeline(TEMPORAL_NN).isValid() || !device.precompilePipeline(BLIT).isValid()
				|| (neuralPipeline != null && !device.precompilePipeline(neuralPipeline).isValid()) || !device.precompilePipeline(SHARPEN).isValid()
				|| !Capture.pipelinesValid()) {
			fail("shaders failed to compile (see log)", null);
			return false;
		}
		pipelinesChecked = true;
		return true;
	}

	private void fail(String reason, Throwable cause) {
		failure = reason;
		levelActive = false;
		frameActive = false;
		LOG.error("MineSR disabled: {}", reason, cause);
		release();
	}
}
