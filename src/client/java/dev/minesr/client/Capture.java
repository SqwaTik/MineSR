package dev.minesr.client;

import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.pipeline.BindGroupLayout;
import com.mojang.blaze3d.pipeline.ColorTargetState;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.FilterMode;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import dev.minesr.capture.CaptureWriter;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.resources.Identifier;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Training-data capture. While active, the world renders into a fixed high-resolution target
 * instead of the screen; a downscaled copy is shown so you can keep playing. In bursts of
 * consecutive frames, color and depth are read back and handed to a writer thread
 * ({@link CaptureWriter}), which saves the target image and the low-res inputs.
 *
 * <p>The GPU is never waited on: readbacks complete asynchronously, and a frame is skipped
 * (leaving a gap the training scripts split on) if all readback buffers are busy.
 */
public final class Capture {
	private static final Logger LOG = LoggerFactory.getLogger("minesr");
	private static final int READBACK_SLOTS = 3;

	private static final RenderPipeline DEPTH_COPY = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/depth_copy"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/depth_copy"))
			.withBindGroupLayout(BindGroupLayout.builder().withSampler("DepthSampler").build())
			.withColorTargetState(new ColorTargetState(Optional.empty(), GpuFormat.R32_FLOAT, ColorTargetState.WRITE_ALL))
			.build();

	private static final RenderPipeline BLIT = RenderPipeline.builder(RenderPipelines.POST_PROCESSING_SNIPPET)
			.withLocation(id("pipeline/blit"))
			.withVertexShader(Identifier.withDefaultNamespace("core/screenquad"))
			.withFragmentShader(id("post/blit"))
			.withBindGroupLayout(BindGroupLayout.builder().withSampler("InSampler").build())
			.withColorTargetState(ColorTargetState.DEFAULT)
			.build();

	private final MineSRConfig config;
	private boolean active;
	private Path session;
	private TextureTarget target;
	private GpuTexture depthColor;
	private GpuTextureView depthColorView;
	private final ArrayDeque<Readback> freeSlots = new ArrayDeque<>();
	private ThreadPoolExecutor cutter;   // raw 16 MB frame -> ~3 MB pieces, fast enough to keep up
	private ThreadPoolExecutor writer;   // compression + disk; catches up between bursts

	private long frame;          // frames rendered while capturing (gaps in saved files = skipped)
	private long burst;
	private int burstRemaining;
	private long nextBurstAt;    // System.nanoTime()
	private long lastFrameNanos;
	private double frameMs;
	private final Matrix4f viewProj = new Matrix4f();
	private Vec3 camera = Vec3.ZERO;

	private final AtomicLong saved = new AtomicLong();
	private final AtomicLong bytes = new AtomicLong();
	private final AtomicLong dropped = new AtomicLong();      // readback buffers busy
	private final AtomicLong writerFull = new AtomicLong();   // disk writer behind
	private final AtomicLong writeNanos = new AtomicLong();
	private volatile String error;

	private record Readback(GpuBuffer color, GpuBuffer depth) {
	}

	Capture(MineSRConfig config) {
		this.config = config;
	}

	private static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath("minesr", path);
	}

	public boolean isActive() {
		return active;
	}

	public RenderTarget target() {
		return target;
	}

	/** Texture derivative scale: pick the mip level the half-size target image would use. */
	public float lodScale() {
		return 2.0f;
	}

	public void start(Path capturesRoot) {
		if (active) {
			return;
		}
		session = capturesRoot.resolve("session-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")));
		cutter = pool("minesr-capture-cut", 1, 6);
		writer = pool("minesr-capture-writer", 3, 96); // a few bursts' worth of small pieces
		frame = 0;
		burst = 0;
		lastFrameNanos = 0;
		burstRemaining = 0;
		nextBurstAt = System.nanoTime() + 2_000_000_000L; // let chunks settle before the first burst
		saved.set(0);
		bytes.set(0);
		dropped.set(0);
		writerFull.set(0);
		writeNanos.set(0);
		error = null;
		active = true;
		LOG.info("MineSR capture started: {}", session);
	}

	private ThreadPoolExecutor pool(String name, int threads, int queue) {
		return new ThreadPoolExecutor(threads, threads, 30, TimeUnit.SECONDS, new ArrayBlockingQueue<>(queue),
				r -> {
					Thread t = new Thread(r, name);
					t.setDaemon(true);
					t.setPriority(Thread.MIN_PRIORITY); // the game comes first
					return t;
				}, (task, executor) -> writerFull.incrementAndGet());
	}

	/** Stops capturing; files still being written finish in the background. */
	public void stop() {
		if (!active) {
			return;
		}
		active = false;
		cutter.shutdown();
		new Thread(() -> { // the writer may only shut down once the cutter has handed over everything
			try {
				cutter.awaitTermination(1, TimeUnit.MINUTES);
			} catch (InterruptedException ignored) {
			}
			writer.shutdown();
		}, "minesr-capture-stop").start();
		release();
		LOG.info("MineSR capture stopped: {} frames, {} MB", saved.get(), bytes.get() >> 20);
	}

	/** Waits for queued frames to be written (tests). */
	public boolean awaitWriter(long seconds) throws InterruptedException {
		return writer == null
				|| (cutter.awaitTermination(seconds, TimeUnit.SECONDS) && writer.awaitTermination(seconds, TimeUnit.SECONDS));
	}

	public String describe() {
		if (!active && session == null) {
			return "not capturing";
		}
		long n = Math.max(1, saved.get());
		return String.format("%s %s: %d frames saved in %d bursts, %d MB, skipped %d (GPU busy) + %d (disk behind), %.0f ms/frame to write%s",
				active ? "capturing to" : "last capture", session.getFileName(), saved.get(), burst, bytes.get() >> 20,
				dropped.get(), writerFull.get(), writeNanos.get() / 1e6 / n, error == null ? "" : ", error: " + error);
	}

	public Path session() {
		return session;
	}

	/** Called where the jitter would go: remember matrices, no jitter (it's applied offline). */
	void beginFrame(Matrix4f projection, Matrix4fc viewRotation, Vec3 cameraPos) {
		ensureTargets();
		viewProj.set(projection).mul(viewRotation);
		camera = cameraPos;
		long now = System.nanoTime();
		frameMs = lastFrameNanos == 0 ? 0.0 : (now - lastFrameNanos) / 1e6;
		lastFrameNanos = now;
	}

	/** After the world pass: show it, and read it back if a burst is running. */
	void endFrame(RenderTarget main) {
		frame++;
		CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
		var samplers = RenderSystem.getSamplerCache();

		try (RenderPass pass = encoder.createRenderPass(() -> "MineSR capture preview", main.getColorTextureView(), Optional.empty())) {
			pass.setPipeline(BLIT);
			RenderSystem.bindDefaultUniforms(pass);
			pass.bindTexture("InSampler", target.getColorTextureView(), samplers.getClampToEdge(FilterMode.LINEAR));
			pass.draw(3, 1, 0, 0);
		}

		long now = System.nanoTime();
		if (burstRemaining == 0 && now >= nextBurstAt && bytes.get() < config.captureMaxMegabytes * (1L << 20)) {
			burstRemaining = config.captureBurstFrames;
			burst++;
		}
		if (burstRemaining == 0) {
			return;
		}
		burstRemaining--;
		if (burstRemaining == 0) {
			nextBurstAt = now + (long) (config.captureBurstIntervalSeconds * 1e9);
		}
		Readback slot = freeSlots.poll();
		if (slot == null) {
			dropped.incrementAndGet(); // GPU readbacks still in flight; skip rather than stall
			return;
		}

		try (RenderPass pass = encoder.createRenderPass(() -> "MineSR capture depth", depthColorView, Optional.empty())) {
			pass.setPipeline(DEPTH_COPY);
			RenderSystem.bindDefaultUniforms(pass);
			pass.bindTexture("DepthSampler", target.getDepthTextureView(), samplers.getClampToEdge(FilterMode.NEAREST));
			pass.draw(3, 1, 0, 0);
		}

		int width = target.width, height = target.height;
		CaptureWriter.Meta meta = new CaptureWriter.Meta(frame, burst, frameMs,
				new double[] {camera.x, camera.y, camera.z}, viewProj.get(new float[16]),
				RenderSystem.getDevice().getDeviceInfo().isZZeroToOne());
		Path dir = session.resolve(String.format("burst-%04d", burst));
		// Both copies are queued in order; the second callback fires after both are done.
		encoder.copyTextureToBuffer(target.getColorTexture(), slot.color(), 0, () -> { }, 0);
		encoder.copyTextureToBuffer(depthColor, slot.depth(), 0, () -> collect(slot, meta, dir, width, height), 0);
	}

	private void collect(Readback slot, CaptureWriter.Meta meta, Path dir, int width, int height) {
		byte[] rgba = new byte[width * height * 4];
		float[] depth = new float[width * height];
		try (GpuBufferSlice.MappedView view = slot.color().map(true, false)) {
			view.data().get(0, rgba);
		}
		try (GpuBufferSlice.MappedView view = slot.depth().map(true, false)) {
			ByteBuffer data = view.data().duplicate().order(ByteOrder.nativeOrder());
			data.asFloatBuffer().get(depth);
		}
		if (active && slot.color().size() == (long) width * height * 4) {
			freeSlots.add(slot);
		} else {
			slot.color().close(); // capture stopped or resized meanwhile
			slot.depth().close();
		}
		if (cutter.isShutdown()) {
			return;
		}
		cutter.execute(() -> {
			CaptureWriter.Prepared prepared = CaptureWriter.prepare(meta, rgba, depth, width, height);
			writer.execute(() -> {
				try {
					long t0 = System.nanoTime();
					bytes.addAndGet(CaptureWriter.write(dir, prepared));
					writeNanos.addAndGet(System.nanoTime() - t0);
					saved.incrementAndGet();
				} catch (Exception e) {
					error = e.toString();
					LOG.error("MineSR capture write failed", e);
				}
			});
		});
	}

	private void ensureTargets() {
		int width = config.captureWidth, height = config.captureHeight;
		if (target != null && target.width == width && target.height == height) {
			return;
		}
		release();
		GpuDevice device = RenderSystem.getDevice();
		target = new TextureTarget("MineSR capture", width, height, true, GpuFormat.RGBA8_UNORM);
		int usage = GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_RENDER_ATTACHMENT | GpuTexture.USAGE_COPY_SRC;
		depthColor = device.createTexture(() -> "MineSR capture depth", usage, GpuFormat.R32_FLOAT, width, height, 1, 1);
		depthColorView = device.createTextureView(depthColor);
		for (int i = 0; i < READBACK_SLOTS; i++) {
			int index = i;
			int bufferUsage = GpuBuffer.USAGE_MAP_READ | GpuBuffer.USAGE_COPY_DST;
			freeSlots.add(new Readback(
					device.createBuffer(() -> "MineSR capture color " + index, bufferUsage, (long) width * height * 4),
					device.createBuffer(() -> "MineSR capture depth " + index, bufferUsage, (long) width * height * 4)));
		}
	}

	private void release() {
		if (target != null) {
			target.destroyBuffers();
			target = null;
		}
		if (depthColorView != null) {
			depthColorView.close();
			depthColorView = null;
		}
		if (depthColor != null) {
			depthColor.close();
			depthColor = null;
		}
		for (Readback slot : freeSlots) {
			slot.color().close();
			slot.depth().close();
		}
		freeSlots.clear(); // slots still in flight are closed by collect()
	}

	static boolean pipelinesValid() {
		GpuDevice device = RenderSystem.getDevice();
		return device.precompilePipeline(DEPTH_COPY).isValid() && device.precompilePipeline(BLIT).isValid();
	}
}
