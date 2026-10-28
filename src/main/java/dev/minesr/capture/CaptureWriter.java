package dev.minesr.capture;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.Deflater;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;

/**
 * Writes one captured frame to disk. Format (documented in training/README.md):
 * <pre>
 * frame-000123.json                 camera, matrices, jitter, sizes
 * frame-000123-target.png           960x540 answer image
 * frame-000123-quality.png          640x360 low-res input
 * frame-000123-quality-depth.f32.gz raw depth, float32 little-endian
 * frame-000123-performance.png / -performance-depth.f32.gz
 * </pre>
 * Images and depth are stored top row first (so they look right in an image viewer).
 */
public final class CaptureWriter {
	/** Everything about the frame that isn't pixels. */
	public record Meta(long frame, long burst, double frameMs, double[] camera, float[] viewProj, boolean zZeroToOne) {
	}

	private CaptureWriter() {
	}

	/** A frame cut into its pieces: small enough (~3 MB) to queue a whole burst. */
	public record Prepared(Meta meta, int width, int height, byte[] target, CaptureFrame.LowRes[] lowRes) {
	}

	/** The CPU-cheap part (a few ms): target image and low-res inputs. */
	public static Prepared prepare(Meta meta, byte[] rgba, float[] depth, int width, int height) {
		CaptureFrame.LowRes[] low = new CaptureFrame.LowRes[CaptureFrame.PRESETS.length];
		for (int i = 0; i < low.length; i++) {
			low[i] = CaptureFrame.lowRes(CaptureFrame.PRESETS[i], rgba, depth, width, height, meta.frame());
		}
		return new Prepared(meta, width, height, CaptureFrame.target(rgba, width, height), low);
	}

	/** @return bytes written */
	public static long write(Path dir, Meta meta, byte[] rgba, float[] depth, int width, int height) throws IOException {
		return write(dir, prepare(meta, rgba, depth, width, height));
	}

	/** The slow part (compression and disk). @return bytes written */
	public static long write(Path dir, Prepared frame) throws IOException {
		Meta meta = frame.meta();
		int width = frame.width(), height = frame.height();
		Files.createDirectories(dir);
		String base = String.format("frame-%06d", meta.frame());
		long bytes = 0;

		JsonObject json = new JsonObject();
		json.addProperty("version", 1);
		json.addProperty("frame", meta.frame());
		json.addProperty("burst", meta.burst());
		json.addProperty("frameMs", meta.frameMs());
		json.add("camera", array(meta.camera()));
		JsonArray vp = new JsonArray();
		for (float v : meta.viewProj()) {
			vp.add(v);
		}
		json.add("viewProjColumnMajor", vp);
		json.addProperty("zZeroToOne", meta.zZeroToOne());
		json.addProperty("captureWidth", width);
		json.addProperty("captureHeight", height);
		json.addProperty("rowOrder", "top-down");

		bytes += png(dir.resolve(base + "-target.png"), frame.target(), width / 2, height / 2);
		JsonObject target = new JsonObject();
		target.addProperty("width", width / 2);
		target.addProperty("height", height / 2);
		json.add("target", target);

		JsonObject presets = new JsonObject();
		for (CaptureFrame.LowRes low : frame.lowRes()) {
			String name = low.preset().name();
			bytes += png(dir.resolve(base + "-" + name + ".png"), low.rgb(), low.width(), low.height());
			bytes += depth(dir.resolve(base + "-" + name + "-depth.f32.gz"), low.depth(), low.width(), low.height());
			JsonObject p = new JsonObject();
			p.addProperty("width", low.width());
			p.addProperty("height", low.height());
			p.addProperty("factor", low.preset().factor());
			JsonArray jitter = new JsonArray();
			jitter.add(low.jitterX());
			jitter.add(low.jitterY());
			p.add("jitter", jitter);
			presets.add(name, p);
		}
		json.add("presets", presets);
		byte[] text = json.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
		Files.write(dir.resolve(base + ".json"), text);
		return bytes + text.length;
	}

	/** RGB bottom-up -> PNG top-down (public entry point for debug snapshots). */
	public static long savePng(Path file, byte[] rgb, int width, int height) throws IOException {
		return png(file, rgb, width, height);
	}

	/** RGB bottom-up -> PNG top-down. */
	static long png(Path file, byte[] rgb, int width, int height) throws IOException {
		byte[] topDown = new byte[rgb.length];
		int stride = width * 3;
		for (int y = 0; y < height; y++) {
			System.arraycopy(rgb, (height - 1 - y) * stride, topDown, y * stride, stride);
		}
		try (OutputStream out = new java.io.BufferedOutputStream(Files.newOutputStream(file), 1 << 16)) {
			FastPng.write(out, topDown, width, height);
		}
		return Files.size(file);
	}

	/** Depth bottom-up -> gzip'd float32 little-endian, top-down. */
	static long depth(Path file, float[] depth, int width, int height) throws IOException {
		ByteBuffer raw = ByteBuffer.allocate(width * height * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (int y = height - 1; y >= 0; y--) {
			for (int x = 0; x < width; x++) {
				raw.putFloat(depth[y * width + x]);
			}
		}
		try (OutputStream out = new FastGzip(Files.newOutputStream(file))) {
			out.write(raw.array());
		}
		return Files.size(file);
	}

	/** gzip at the fastest level. */
	private static final class FastGzip extends GZIPOutputStream {
		FastGzip(OutputStream out) throws IOException {
			super(out, 1 << 16);
			def.setLevel(Deflater.BEST_SPEED);
		}
	}

	private static JsonArray array(double[] values) {
		JsonArray a = new JsonArray();
		for (double v : values) {
			a.add(v);
		}
		return a;
	}
}
