package dev.minesr.capture;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/**
 * Minimal 8-bit RGB PNG encoder: "Sub" row filter and Huffman-only deflate. ImageIO's encoder took
 * ~60 ms for a 960x540 image, far too slow to keep up with a capture burst; this is several
 * times faster and still lossless and readable by any PNG decoder.
 */
public final class FastPng {
	private static final byte[] SIGNATURE = {(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

	private FastPng() {
	}

	/** @param rgb top row first, 3 bytes per pixel */
	public static void write(OutputStream out, byte[] rgb, int width, int height) throws IOException {
		DataOutputStream data = new DataOutputStream(out);
		data.write(SIGNATURE);

		ByteArrayOutputStream header = new ByteArrayOutputStream();
		DataOutputStream h = new DataOutputStream(header);
		h.writeInt(width);
		h.writeInt(height);
		h.writeByte(8);  // bit depth
		h.writeByte(2);  // color type: RGB
		h.writeByte(0);  // compression
		h.writeByte(0);  // filter method
		h.writeByte(0);  // no interlace
		chunk(data, "IHDR", header.toByteArray());

		ByteArrayOutputStream compressed = new ByteArrayOutputStream(rgb.length / 2);
		Deflater deflater = new Deflater(Deflater.BEST_SPEED);
		deflater.setStrategy(Deflater.HUFFMAN_ONLY); // several times faster; the Sub filter already removes most redundancy
		try (DeflaterOutputStream z = new DeflaterOutputStream(compressed, deflater, 1 << 16)) {
			int stride = width * 3;
			byte[] row = new byte[stride + 1];
			row[0] = 1; // Sub: each byte minus the same channel of the pixel to its left
			for (int y = 0; y < height; y++) {
				int base = y * stride;
				for (int i = 0; i < stride; i++) {
					int left = i >= 3 ? rgb[base + i - 3] : 0;
					row[i + 1] = (byte) (rgb[base + i] - left);
				}
				z.write(row);
			}
		} finally {
			deflater.end();
		}
		chunk(data, "IDAT", compressed.toByteArray());
		chunk(data, "IEND", new byte[0]);
		data.flush();
	}

	private static void chunk(DataOutputStream out, String type, byte[] body) throws IOException {
		byte[] name = type.getBytes(StandardCharsets.US_ASCII);
		CRC32 crc = new CRC32();
		crc.update(name);
		crc.update(body);
		out.writeInt(body.length);
		out.write(name);
		out.write(body);
		out.writeInt((int) crc.getValue());
	}
}
