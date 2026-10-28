package dev.minesr;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JitterTest {
	@Test
	void haltonMatchesKnownValues() {
		assertEquals(0.5, Jitter.halton(1, 2), 1e-12);
		assertEquals(0.25, Jitter.halton(2, 2), 1e-12);
		assertEquals(0.75, Jitter.halton(3, 2), 1e-12);
		assertEquals(1.0 / 3.0, Jitter.halton(1, 3), 1e-12);
		assertEquals(2.0 / 3.0, Jitter.halton(2, 3), 1e-12);
		assertEquals(1.0 / 9.0, Jitter.halton(3, 3), 1e-12);
	}

	@Test
	void moreUpscalingNeedsMorePhases() {
		assertEquals(8, Jitter.phaseCount(1920, 1920));
		assertEquals(18, Jitter.phaseCount(1280, 1920));
		assertEquals(32, Jitter.phaseCount(960, 1920));
	}

	@Test
	void offsetsStayWithinHalfAPixelAndDontRepeatWithinAPeriod() {
		int phases = 32;
		Set<String> seen = new HashSet<>();
		for (int frame = 0; frame < phases; frame++) {
			float[] o = Jitter.offset(frame, phases);
			assertTrue(o[0] >= -0.5f && o[0] < 0.5f && o[1] >= -0.5f && o[1] < 0.5f);
			assertTrue(seen.add(o[0] + "," + o[1]), "repeated offset at frame " + frame);
		}
		assertEquals(Jitter.offset(0, phases)[0], Jitter.offset(phases, phases)[0]);
	}

	@Test
	void gridOffsetsMatchTheTrainingData() {
		// Same positions CaptureFrame uses when cutting low-res inputs out of a capture.
		for (int factor : new int[] {3, 4}) {
			for (long frame = 0; frame < 64; frame++) {
				dev.minesr.capture.CaptureFrame.Preset preset = new dev.minesr.capture.CaptureFrame.Preset("t", factor);
				byte[] rgba = new byte[24 * 24 * 4];
				var low = dev.minesr.capture.CaptureFrame.lowRes(preset, rgba, new float[24 * 24], 24, 24, frame);
				float[] grid = Jitter.gridOffset(frame, factor);
				assertEquals(low.jitterX(), grid[0], 1e-6);
				assertEquals(low.jitterY(), grid[1], 1e-6);
			}
		}
	}

	@Test
	void presetsMatchFsrScaleFactors() {
		assertEquals(1280, Quality.scaled(1920, Quality.QUALITY.scale));
		assertEquals(960, Quality.scaled(1920, Quality.PERFORMANCE.scale));
		assertEquals(1, Quality.scaled(1, Quality.PERFORMANCE.scale));
	}
}
