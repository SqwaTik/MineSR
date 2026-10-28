package dev.minesr.client;

import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.pipeline.BindGroupLayout;
import com.mojang.blaze3d.pipeline.ColorTargetState;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.GpuSurface;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.systems.SurfaceException;
import com.mojang.blaze3d.textures.FilterMode;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.resources.Identifier;
import org.joml.Vector4f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

/**
 * Frame generation: for every rendered frame N, an extra frame halfway between N-1 and N is
 * shown first, then N half a frame later. Shown frames per second roughly double; the game
 * itself (and input) still runs at the rendered rate, plus about half a frame of delay.
 *
 * <p>Per frame (driven by the mixins):
 * <ol>
 *   <li>After the world pass: hand, screen effects and GUI are redirected onto a transparent
 *       overlay layer, so they are never warped by world motion.</li>
 *   <li>At present time: world + overlay make the real frame N; world N-1, world N and the
 *       upscaler's motion make the generated frame, which is presented instead.</li>
 *   <li>During the next frame, once half a frame interval has passed, the real frame N is
 *       presented ({@link #maybePresentPending}).</li>
 * </ol>
 */
public final class FrameGen {
	private static final Logger LOG = LoggerFactory.getLogger("minesr");

	private static final RenderPipeline COMPOSITE = pipeline("composite", "WorldSampler", "OverlaySampler");
	private static final RenderPipeline INTERPOLATE = pipeline("interpolate", "PrevSampler", "CurSampler", "MotionSampler", "OverlaySampler");

	private static RenderPipeline pipeline(String name, String... samplers) {
		BindGroupLayout.Builder layout = BindGroupLayout.builder();
		for (String s : samplers) {
			layout.withSampler(s);
		}
		return RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
				.withLocation(Identifier.fromNamespaceAndPath("minesr", "pipeline/" + name))
				.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
				.withFragmentShader(Identifier.fromNamespaceAndPath("minesr", "post/" + name))
				.withBindGroupLayout(layout.build())
				.withColorTargetState(ColorTargetState.DEFAULT)
				.build();
	}

	private final MineSRConfig config;
	private final Upscaler upscaler;

	private TextureTarget overlay;
	private final GpuTexture[] world = new GpuTexture[2];       // world images N-1 and N
	private final GpuTextureView[] worldView = new GpuTextureView[2];
	private final GpuTexture[] real = new GpuTexture[2];        // composited real frames (one may be pending)
	private final GpuTextureView[] realView = new GpuTextureView[2];
	private GpuTexture generated;
	private GpuTextureView generatedView;
	private int width, height;
	private int cur;              // index of world N / real N
	private boolean havePrevious;
	private boolean overlayActive;
	private boolean pending;      // real frame N still to be presented
	private int pendingIndex;
	private long generatedAt;     // System.nanoTime() of the last generated present
	private long lastFrameEnd;
	private double frameInterval = 1e9 / 60.0;
	private String failure;
	private boolean pipelinesChecked;

	private long presents;        // everything that reached the screen, real and generated
	private long generatedPresents;
	/** Present times (System.nanoTime) while a pacing measurement runs; null otherwise. */
	private java.util.List<Long> presentTimes;
	/** Debug: next generated frame, save previous world, generated and real frame as PNGs here. */
	private java.nio.file.Path snapshotDir;

	public void startPresentTiming() {
		presentTimes = new java.util.ArrayList<>();
	}

	public java.util.List<Long> stopPresentTiming() {
		java.util.List<Long> out = presentTimes == null ? java.util.List.of() : presentTimes;
		presentTimes = null;
		return out;
	}

	public void requestSnapshot(java.nio.file.Path dir) {
		snapshotDir = dir;
	}

	private void markPresent() {
		presents++;
		if (presentTimes != null) {
			presentTimes.add(System.nanoTime());
		}
	}

	private void saveTexture(GpuTexture texture, java.nio.file.Path file) {
		int w = width, h = height;
		var buffer = RenderSystem.getDevice().createBuffer(() -> "MineSR snapshot", 9, (long) w * h * 4);
		RenderSystem.getDevice().createCommandEncoder().copyTextureToBuffer(texture, buffer, 0, () -> {
			byte[] rgb = new byte[w * h * 3];
			try (var view = buffer.map(true, false)) {
				for (int i = 0; i < w * h; i++) {
					rgb[i * 3] = view.data().get(i * 4);
					rgb[i * 3 + 1] = view.data().get(i * 4 + 1);
					rgb[i * 3 + 2] = view.data().get(i * 4 + 2);
				}
			}
			buffer.close();
			try {
				java.nio.file.Files.createDirectories(file.getParent());
				dev.minesr.capture.CaptureWriter.savePng(file, rgb, w, h);
			} catch (java.io.IOException e) {
				LOG.warn("snapshot failed: {}", e.toString());
			}
		}, 0);
	}

	FrameGen(MineSRConfig config, Upscaler upscaler) {
		this.config = config;
		this.upscaler = upscaler;
	}

	public boolean enabled() {
		return config.frameGeneration && failure == null;
	}

	public boolean isOverlayActive() {
		return overlayActive;
	}

	public RenderTarget overlayTarget() {
		return overlay;
	}

	public long presentCount() {
		return presents;
	}

	public long generatedCount() {
		return generatedPresents;
	}

	public String describe() {
		if (failure != null) {
			return "frame generation disabled: " + failure;
		}
		return config.frameGeneration ? "frame generation on" : "frame generation off";
	}

	/** After the world pass: everything drawn from here until present goes onto the overlay. */
	public void beginOverlay(RenderTarget main) {
		if (!enabled() || !upscaler.motionReadyThisFrame() || !checkPipelines()) {
			havePrevious = havePrevious && overlay != null && enabled();
			return;
		}
		ensureTargets(main.width, main.height);
		RenderSystem.getDevice().createCommandEncoder()
				.clearColorAndDepthTextures(overlay.getColorTexture(), new Vector4f(0.0f), overlay.getDepthTexture(), 0.0);
		overlayActive = true;
	}

	/** Vanilla clears the main depth buffer before drawing the GUI; do the same for the overlay. */
	public void beforeGui() {
		if (overlayActive) {
			RenderSystem.getDevice().createCommandEncoder().clearDepthTexture(overlay.getDepthTexture(), 0.0);
		}
	}

	/**
	 * Replaces what vanilla blits to the screen at the end of the frame. Returns the texture
	 * to show now: the generated frame (and the real one becomes pending), or the real frame
	 * when there is nothing to interpolate from.
	 */
	/**
	 * Ends the overlay phase before vanilla looks up the main image for presenting; otherwise
	 * mainRenderTarget() would still be redirected to the overlay.
	 */
	public boolean endOverlay() {
		boolean was = overlayActive;
		overlayActive = false;
		overlayEnded = was;
		return was;
	}

	private boolean overlayEnded;

	public GpuTextureView beforePresent(GpuSurface surface, GpuTextureView mainColor, RenderTarget main) {
		long now = System.nanoTime();
		if (lastFrameEnd != 0) {
			frameInterval += ((now - lastFrameEnd) - frameInterval) * 0.1;
		}
		lastFrameEnd = now;
		markPresent();
		boolean overlayUsed = overlayEnded || overlayActive;
		overlayEnded = false;
		overlayActive = false;
		mainColor = main.getColorTextureView(); // the world image, never the overlay
		if (!overlayUsed) {
			if (pending) {
				presentPending(surface); // frame generation just turned off: don't drop the frame
			}
			return mainColor;
		}
		try {
			CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
			int prev = cur;
			cur ^= 1;
			encoder.copyTextureToTexture(main.getColorTexture(), world[cur], 0, 0, 0, 0, 0, width, height);
			int realIndex = pending && pendingIndex == cur ? cur ^ 1 : cur; // never overwrite a frame still waiting
			draw(encoder, COMPOSITE, realView[realIndex], pass -> {
				pass.bindTexture("WorldSampler", worldView[cur], nearest());
				pass.bindTexture("OverlaySampler", overlay.getColorTextureView(), nearest());
			});
			boolean generate = havePrevious && upscaler.continuousWithPreviousFrame();
			havePrevious = true;
			if (pending) {
				presentPending(surface); // pacing hooks didn't get to it; show it now, late rather than never
			}
			if (!generate) {
				return realView[realIndex];
			}
			draw(encoder, INTERPOLATE, generatedView, pass -> {
				pass.bindTexture("PrevSampler", worldView[prev], linear());
				pass.bindTexture("CurSampler", worldView[cur], linear());
				pass.bindTexture("MotionSampler", upscaler.motionView(), nearest());
				pass.bindTexture("OverlaySampler", overlay.getColorTextureView(), nearest());
			});
			if (snapshotDir != null) {
				saveTexture(world[prev], snapshotDir.resolve("1-previous.png"));
				saveTexture(generated, snapshotDir.resolve("2-generated.png"));
				saveTexture(real[realIndex], snapshotDir.resolve("3-current.png"));
				snapshotDir = null;
			}
			pending = true;
			pendingIndex = realIndex;
			generatedAt = now;
			generatedPresents++;
			return generatedView;
		} catch (RuntimeException e) {
			fail("frame generation failed: " + e, e);
			return mainColor;
		}
	}

	/** Called at a few points during the next frame: present the real frame once half an interval has passed. */
	public void maybePresentPending(GpuSurface surface) {
		maybePresentPending(surface, false);
	}

	/**
	 * @param lastChance the last check point of the frame (before the GUI). If the real frame isn't
	 *     due yet, wait for its moment here: otherwise it would go out at the end of the frame,
	 *     right next to the generated one, and the two would look like one.
	 */
	public void maybePresentPending(GpuSurface surface, boolean lastChance) {
		if (!pending) {
			return;
		}
		long due = generatedAt + (long) (frameInterval * 0.5);
		long now = System.nanoTime();
		// Waiting here for the exact moment was tried: it didn't even out the spacing (what reaches
		// the screen when is decided by the GPU queue, not by these CPU-side check points) and it
		// cost rendered FPS. Even pacing needs presenting from the GPU timeline; for now, present
		// at the first check point that is due.
		if (now < due) {
			return;
		}
		presentPending(surface);
	}

	private void presentPending(GpuSurface surface) {
		pending = false;
		if (!surface.isAcquired()) {
			return;
		}
		CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
		surface.blitFromTexture(encoder, realView[pendingIndex]);
		encoder.submit();
		surface.present();
		markPresent();
		try {
			surface.acquireNextTexture(); // the rest of this frame presents into a fresh image
		} catch (SurfaceException e) {
			LOG.warn("MineSR frame generation: could not acquire the next screen image: {}", e.getMessage());
		}
	}

	private interface Binder {
		void bind(RenderPass pass);
	}

	private void draw(CommandEncoder encoder, RenderPipeline pipeline, GpuTextureView target, Binder binder) {
		try (RenderPass pass = encoder.createRenderPass(() -> "MineSR frame generation", target, Optional.empty())) {
			pass.setPipeline(pipeline);
			RenderSystem.bindDefaultUniforms(pass);
			binder.bind(pass);
			pass.draw(3, 1, 0, 0);
		}
	}

	private static com.mojang.blaze3d.textures.GpuSampler nearest() {
		return RenderSystem.getSamplerCache().getClampToEdge(FilterMode.NEAREST);
	}

	private static com.mojang.blaze3d.textures.GpuSampler linear() {
		return RenderSystem.getSamplerCache().getClampToEdge(FilterMode.LINEAR);
	}

	private void ensureTargets(int w, int h) {
		if (overlay != null && width == w && height == h) {
			return;
		}
		release();
		GpuDevice device = RenderSystem.getDevice();
		overlay = new TextureTarget("MineSR overlay", w, h, true, GpuFormat.RGBA8_UNORM);
		int usage = GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_RENDER_ATTACHMENT
				| GpuTexture.USAGE_COPY_DST | GpuTexture.USAGE_COPY_SRC;
		for (int i = 0; i < 2; i++) {
			int index = i;
			world[i] = device.createTexture(() -> "MineSR world " + index, usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1);
			worldView[i] = device.createTextureView(world[i]);
			real[i] = device.createTexture(() -> "MineSR real frame " + index, usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1);
			realView[i] = device.createTextureView(real[i]);
		}
		generated = device.createTexture(() -> "MineSR generated frame", usage, GpuFormat.RGBA8_UNORM, w, h, 1, 1);
		generatedView = device.createTextureView(generated);
		width = w;
		height = h;
		havePrevious = false;
		pending = false;
	}

	/** Frees GPU resources; the next frame with frame generation on reallocates. */
	public void release() {
		if (overlay != null) {
			overlay.destroyBuffers();
			overlay = null;
		}
		for (int i = 0; i < 2; i++) {
			if (worldView[i] != null) worldView[i].close();
			if (world[i] != null) world[i].close();
			if (realView[i] != null) realView[i].close();
			if (real[i] != null) real[i].close();
			worldView[i] = null;
			world[i] = null;
			realView[i] = null;
			real[i] = null;
		}
		if (generatedView != null) generatedView.close();
		if (generated != null) generated.close();
		generatedView = null;
		generated = null;
		havePrevious = false;
		pending = false;
		overlayActive = false;
	}

	private boolean checkPipelines() {
		if (pipelinesChecked) {
			return true;
		}
		GpuDevice device = RenderSystem.getDevice();
		if (!device.precompilePipeline(COMPOSITE).isValid() || !device.precompilePipeline(INTERPOLATE).isValid()) {
			fail("shaders failed to compile (see log)", null);
			return false;
		}
		pipelinesChecked = true;
		return true;
	}

	public void retry() {
		failure = null;
		pipelinesChecked = false;
	}

	private void fail(String reason, Throwable cause) {
		failure = reason;
		LOG.error("MineSR frame generation disabled: {}", reason, cause);
		release();
	}
}
