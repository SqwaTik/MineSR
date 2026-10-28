package dev.minesr.capture;

import dev.minesr.Jitter;

/**
 * Turns one high-resolution frame into training samples.
 *
 * <p>The world is rendered at {@code W x H} (both divisible by 12). From that:
 * <ul>
 *   <li><b>target</b> ({@code W/2 x H/2}): the answer the network should produce, each pixel the
 *       average of 2x2 high-res samples (4x supersampling).</li>
 *   <li><b>low-res inputs</b> for each preset: one high-res sample out of every {@code f x f}
 *       block, at a per-frame offset. That is exactly what rendering at {@code W/f} with a
 *       sub-pixel jitter produces, so these match what the upscaler sees in game.
 *       Quality: f = 3 (640x360 for a 1920x1080 capture, 1.5x upscale). Performance: f = 4 (2x).</li>
 * </ul>
 *
 * <p>All arrays are bottom row first, as the GPU returns them.
 */
public final class CaptureFrame {
	/** Preset name and its high-res-to-low-res factor; target is always half the capture size. */
	public record Preset(String name, int factor) {
	}

	public static final Preset[] PRESETS = {new Preset("quality", 3), new Preset("performance", 4)};

	/** One low-res input: RGB bytes, raw depth, and the jitter it was taken with. */
	public record LowRes(Preset preset, int width, int height, byte[] rgb, float[] depth, float jitterX, float jitterY) {
	}

	private CaptureFrame() {
	}

	/** 2x2 box average of RGBA8 into RGB8. */
	public static byte[] target(byte[] rgba, int width, int height) {
		int tw = width / 2, th = height / 2;
		byte[] out = new byte[tw * th * 3];
		for (int y = 0; y < th; y++) {
			for (int x = 0; x < tw; x++) {
				for (int c = 0; c < 3; c++) {
					int sum = 0;
					for (int dy = 0; dy < 2; dy++) {
						for (int dx = 0; dx < 2; dx++) {
							sum += rgba[((2 * y + dy) * width + 2 * x + dx) * 4 + c] & 0xFF;
						}
					}
					out[(y * tw + x) * 3 + c] = (byte) ((sum + 2) / 4);
				}
			}
		}
		return out;
	}

	/**
	 * Point-samples one high-res pixel per {@code f x f} block. The offset inside the block walks
	 * a Halton sequence over frames, like the in-game jitter.
	 */
	public static LowRes lowRes(Preset preset, byte[] rgba, float[] depth, int width, int height, long frame) {
		int f = preset.factor();
		int lw = width / f, lh = height / f;
		int i = (int) (frame % 64) + 1;
		int ax = Math.min(f - 1, (int) (Jitter.halton(i, 2) * f));
		int ay = Math.min(f - 1, (int) (Jitter.halton(i, 3) * f));
		byte[] rgb = new byte[lw * lh * 3];
		float[] d = new float[lw * lh];
		for (int y = 0; y < lh; y++) {
			for (int x = 0; x < lw; x++) {
				int src = (f * y + ay) * width + f * x + ax;
				rgb[(y * lw + x) * 3] = rgba[src * 4];
				rgb[(y * lw + x) * 3 + 1] = rgba[src * 4 + 1];
				rgb[(y * lw + x) * 3 + 2] = rgba[src * 4 + 2];
				d[y * lw + x] = depth[src];
			}
		}
		return new LowRes(preset, lw, lh, rgb, d, jitter(ax, f), jitter(ay, f));
	}

	/**
	 * In the upscaler's convention, low-res texel i shows the scene at {@code i + 0.5 - J}
	 * (low-res pixel units). Sampling high-res pixel {@code f*i + a} shows it at
	 * {@code i + (a + 0.5) / f}, so {@code J = 0.5 - (a + 0.5) / f}.
	 */
	static float jitter(int offset, int factor) {
		return 0.5f - (offset + 0.5f) / factor;
	}
}
