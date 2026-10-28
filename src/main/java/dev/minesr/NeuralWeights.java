package dev.minesr;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;

/**
 * The trained network (training/train_upscaler.py, exported as upscaler_net.json), packed into
 * the {@code NetWeights} uniform block of temporal_nn.fsh, for H hidden units per layer:
 * <pre>
 * vec4 W1[H * 5]   H rows x 20 inputs (17 used, rest zero)
 * vec4 B1[H / 4]
 * vec4 W2[H * H/4] H rows x H
 * vec4 B2[H / 4]
 * vec4 W3[H]       4 rows x H
 * vec4 B3          4 biases
 * </pre>
 * Row r of a layer with k (padded) inputs occupies floats [r*k, (r+1)*k).
 *
 * @param hidden units per hidden layer; the shader is compiled with {@code #define HIDDEN hidden}
 */
public record NeuralWeights(int hidden, float[] data) {
	public static final int FEATURES = 17, OUTPUTS = 4;
	static final int IN_PADDED = 20;

	public static int floats(int hidden) {
		return hidden * IN_PADDED + hidden + hidden * hidden + hidden + OUTPUTS * hidden + OUTPUTS;
	}

	public static NeuralWeights pack(JsonObject json) throws IOException {
		JsonArray layers = json.getAsJsonArray("layers");
		// Version 2: the network decides per low-res pixel (decide.fsh). Version 1 networks were
		// trained for per-output-pixel decisions and would behave differently here.
		if (json.get("version").getAsInt() != 2 || layers.size() != 3
				|| json.get("features").getAsInt() != FEATURES || json.get("outputs").getAsInt() != OUTPUTS) {
			throw new IOException("network file doesn't match this version of the shader");
		}
		int hidden = layers.get(0).getAsJsonObject().getAsJsonArray("bias").size();
		if (hidden < 4 || hidden > 32 || hidden % 4 != 0) {
			throw new IOException("hidden layer size " + hidden + " must be a multiple of 4 between 4 and 32");
		}
		float[] out = new float[floats(hidden)];
		int w1 = 0, b1 = w1 + hidden * IN_PADDED, w2 = b1 + hidden, b2 = w2 + hidden * hidden, w3 = b2 + hidden, b3 = w3 + OUTPUTS * hidden;
		layer(layers.get(0).getAsJsonObject(), hidden, FEATURES, IN_PADDED, out, w1, b1);
		layer(layers.get(1).getAsJsonObject(), hidden, hidden, hidden, out, w2, b2);
		layer(layers.get(2).getAsJsonObject(), OUTPUTS, hidden, hidden, out, w3, b3);
		return new NeuralWeights(hidden, out);
	}

	private static void layer(JsonObject layer, int rows, int inputs, int padded, float[] out, int wOff, int bOff)
			throws IOException {
		JsonArray w = layer.getAsJsonArray("weight");
		JsonArray b = layer.getAsJsonArray("bias");
		if (w.size() != rows || b.size() != rows) {
			throw new IOException("layer has " + w.size() + " rows, expected " + rows);
		}
		for (int r = 0; r < rows; r++) {
			JsonArray row = w.get(r).getAsJsonArray();
			if (row.size() != inputs) {
				throw new IOException("layer row has " + row.size() + " inputs, expected " + inputs);
			}
			for (int c = 0; c < inputs; c++) {
				out[wOff + r * padded + c] = row.get(c).getAsFloat();
			}
			out[bOff + r] = b.get(r).getAsFloat();
		}
	}

	/** The shader's network evaluation on the CPU, reading the packed layout. For tests. */
	public float[] forward(float[] features) {
		int h = hidden;
		int w1 = 0, b1 = w1 + h * IN_PADDED, w2 = b1 + h, b2 = w2 + h * h, w3 = b2 + h, b3 = w3 + OUTPUTS * h;
		float[] in = new float[IN_PADDED];
		System.arraycopy(features, 0, in, 0, FEATURES);
		float[] h1 = dense(w1, b1, h, IN_PADDED, in, true);
		float[] h2 = dense(w2, b2, h, h, h1, true);
		return dense(w3, b3, OUTPUTS, h, h2, false);
	}

	private float[] dense(int wOff, int bOff, int rows, int k, float[] in, boolean relu) {
		float[] out = new float[rows];
		for (int r = 0; r < rows; r++) {
			float s = data[bOff + r];
			for (int c = 0; c < k; c++) {
				s += data[wOff + r * k + c] * in[c];
			}
			out[r] = relu ? Math.max(0.0f, s) : s;
		}
		return out;
	}
}
