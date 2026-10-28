package dev.minesr.capture;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CaptureFrameTest {
	private static final int W = 24, H = 12;

	/** Red channel = x, green = y, depth = y * W + x: every pixel identifies its own position. */
	private static byte[] coordImage() {
		byte[] rgba = new byte[W * H * 4];
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				int i = (y * W + x) * 4;
				rgba[i] = (byte) x;
				rgba[i + 1] = (byte) y;
				rgba[i + 2] = 7;
				rgba[i + 3] = (byte) 255;
			}
		}
		return rgba;
	}

	private static float[] coordDepth() {
		float[] d = new float[W * H];
		for (int i = 0; i < d.length; i++) {
			d[i] = i;
		}
		return d;
	}

	@Test
	void targetAveragesTwoByTwo() {
		byte[] t = CaptureFrame.target(coordImage(), W, H);
		// Target pixel (3, 2) averages x in {6, 7} -> 6.5 -> rounds to 7, y in {4, 5} -> 5.
		int i = (2 * (W / 2) + 3) * 3;
		assertEquals(7, t[i] & 0xFF);
		assertEquals(5, t[i + 1] & 0xFF);
		assertEquals(7, t[i + 2] & 0xFF);
	}

	@Test
	void lowResSamplesMatchTheirJitter() {
		for (CaptureFrame.Preset preset : CaptureFrame.PRESETS) {
			Set<String> offsets = new HashSet<>();
			for (long frame = 0; frame < 32; frame++) {
				CaptureFrame.LowRes low = CaptureFrame.lowRes(preset, coordImage(), coordDepth(), W, H, frame);
				int f = preset.factor();
				assertEquals(W / f, low.width());
				// The upscaler assumes texel i shows the scene at i + 0.5 - J (low-res units).
				// Map that back to high-res pixel units and it must be the center of the sampled pixel.
				for (int y = 0; y < low.height(); y++) {
					for (int x = 0; x < low.width(); x++) {
						int i = y * low.width() + x;
						int srcX = low.rgb()[i * 3] & 0xFF, srcY = low.rgb()[i * 3 + 1] & 0xFF;
						assertEquals((x + 0.5 - low.jitterX()) * f, srcX + 0.5, 1e-4);
						assertEquals((y + 0.5 - low.jitterY()) * f, srcY + 0.5, 1e-4);
						assertEquals(srcY * W + srcX, low.depth()[i], 0.0f);
					}
				}
				assertTrue(Math.abs(low.jitterX()) < 0.5f && Math.abs(low.jitterY()) < 0.5f);
				offsets.add(low.jitterX() + "," + low.jitterY());
			}
			assertEquals(preset.factor() * preset.factor(), offsets.size(), "every sub-pixel position gets used");
		}
	}

	@Test
	void fastPngDecodesExactlyWithJavasReader(@TempDir Path dir) throws Exception {
		int w = 37, h = 11; // odd sizes on purpose
		byte[] rgb = new byte[w * h * 3];
		new java.util.Random(3).nextBytes(rgb);
		Path file = dir.resolve("x.png");
		try (var out = Files.newOutputStream(file)) {
			FastPng.write(out, rgb, w, h);
		}
		BufferedImage img = ImageIO.read(file.toFile());
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int i = (y * w + x) * 3;
				int expected = ((rgb[i] & 0xFF) << 16) | ((rgb[i + 1] & 0xFF) << 8) | (rgb[i + 2] & 0xFF);
				assertEquals(expected, img.getRGB(x, y) & 0xFFFFFF, "pixel " + x + "," + y);
			}
		}
	}

	@Test
	void writerStoresTopDownImagesAndLittleEndianDepth(@TempDir Path dir) throws Exception {
		CaptureWriter.Meta meta = new CaptureWriter.Meta(5, 1, 16.6, new double[] {1, 2, 3}, new float[16], true);
		CaptureWriter.write(dir, meta, coordImage(), coordDepth(), W, H);

		BufferedImage target = ImageIO.read(dir.resolve("frame-000005-target.png").toFile());
		assertEquals(W / 2, target.getWidth());
		// Top row of the file is the bottom-up buffer's last row: y = 10..11 -> 10.5 -> rounds to 11.
		assertEquals(11, (target.getRGB(0, 0) >> 8) & 0xFF);

		try (DataInputStream in = new DataInputStream(new GZIPInputStream(
				Files.newInputStream(dir.resolve("frame-000005-quality-depth.f32.gz"))))) {
			float first = Float.intBitsToFloat(Integer.reverseBytes(in.readInt()));
			// First stored value = top-left of the quality image = highest low-res row.
			CaptureFrame.LowRes low = CaptureFrame.lowRes(CaptureFrame.PRESETS[0], coordImage(), coordDepth(), W, H, 5);
			assertEquals(low.depth()[(low.height() - 1) * low.width()], first, 0.0f);
		}
		assertTrue(Files.readString(dir.resolve("frame-000005.json")).contains("\"jitter\""));
	}
}
