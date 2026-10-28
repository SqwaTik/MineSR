package dev.minesr;

/**
 * Sub-pixel camera offsets. Each frame the low-res image is rendered shifted by a different
 * fraction of a pixel, so over several frames every output pixel gets sampled at many
 * positions. That is what lets a temporal upscaler recover detail a single low-res frame
 * doesn't have.
 */
public final class Jitter {
	private Jitter() {
	}

	/** Radical inverse in the given base: an evenly spread sequence in [0, 1). */
	public static double halton(int index, int base) {
		double f = 1.0, r = 0.0;
		while (index > 0) {
			f /= base;
			r += f * (index % base);
			index /= base;
		}
		return r;
	}

	/**
	 * Sequence length: more upscaling needs more distinct positions per output pixel.
	 * 8 per pixel of the output grid, as in AMD's published FSR 2 guidance.
	 */
	public static int phaseCount(int lowWidth, int outWidth) {
		double ratio = (double) outWidth / lowWidth;
		return Math.max(8, (int) Math.ceil(8.0 * ratio * ratio));
	}

	/**
	 * Offset on a fixed f x f grid of sub-pixel positions, visited in Halton order: exactly what
	 * the training data contains (CaptureFrame cuts low-res inputs from a capture f times larger).
	 * The network was trained on these positions only, so the game uses the same ones.
	 *
	 * @param factor 3 for quality (1.5x upscale), 4 for performance (2x)
	 */
	public static float[] gridOffset(long frame, int factor) {
		int i = (int) (frame % 64) + 1;
		int ax = Math.min(factor - 1, (int) (halton(i, 2) * factor));
		int ay = Math.min(factor - 1, (int) (halton(i, 3) * factor));
		return new float[] {0.5f - (ax + 0.5f) / factor, 0.5f - (ay + 0.5f) / factor};
	}

	/** The grid matching a render scale: 4x4 at 2x upscale and beyond, otherwise 3x3. */
	public static int gridFactor(int lowWidth, int outWidth) {
		return (double) outWidth / lowWidth >= 1.9 ? 4 : 3;
	}

	/** Offset in low-res pixels, in [-0.5, 0.5), for the given frame. */
	public static float[] offset(long frame, int phases) {
		int i = (int) (frame % phases) + 1; // index 0 is (0, 0) in both bases; skip it
		return new float[] {(float) (halton(i, 2) - 0.5), (float) (halton(i, 3) - 0.5)};
	}
}
