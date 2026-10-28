package dev.minesr;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The shipped network, packed the way temporal_nn.fsh reads it, must compute what PyTorch
 * computed for the same inputs (fixture from training/export_parity.py).
 */
class NeuralWeightsTest {
	private static JsonObject resource(String path) throws IOException {
		try (InputStream in = Objects.requireNonNull(NeuralWeightsTest.class.getResourceAsStream(path), path)) {
			return JsonParser.parseString(new String(in.readAllBytes(), StandardCharsets.UTF_8)).getAsJsonObject();
		}
	}

	@Test
	void packedLayoutMatchesPyTorch() throws IOException {
		NeuralWeights net = NeuralWeights.pack(resource("/neural/upscaler_net.json"));
		int h = net.hidden();
		// Sizes of the shader's NetWeights block, in vec4s: W1, B1, W2, B2, W3, B3.
		int vec4s = h * 5 + h / 4 + h * h / 4 + h / 4 + h + 1;
		assertEquals(vec4s * 4, net.data().length, "must fill the shader's NetWeights block exactly");
		JsonObject parity = resource("/neural/parity.json");
		JsonArray inputs = parity.getAsJsonArray("inputs");
		JsonArray outputs = parity.getAsJsonArray("outputs");
		for (int i = 0; i < inputs.size(); i++) {
			JsonArray in = inputs.get(i).getAsJsonArray();
			float[] x = new float[NeuralWeights.FEATURES];
			for (int k = 0; k < x.length; k++) {
				x[k] = in.get(k).getAsFloat();
			}
			float[] y = net.forward(x);
			for (int k = 0; k < NeuralWeights.OUTPUTS; k++) {
				assertEquals(outputs.get(i).getAsJsonArray().get(k).getAsFloat(), y[k], 1e-4, "case " + i + " output " + k);
			}
		}
	}

	@Test
	void rejectsNetworksOfTheWrongShape() throws IOException {
		JsonObject net = resource("/neural/upscaler_net.json");
		net.addProperty("features", 18);
		assertThrows(IOException.class, () -> NeuralWeights.pack(net));
	}
}
