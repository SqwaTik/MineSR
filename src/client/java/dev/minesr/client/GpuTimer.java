package dev.minesr.client;

import com.mojang.blaze3d.systems.GpuQueryPool;
import com.mojang.blaze3d.systems.RenderSystem;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;

/**
 * GPU time of the world pass and of each upscale pass, measured with timestamp queries.
 * Results are read a few frames later so the CPU never waits for the GPU.
 *
 * <p>Comparing GPU milliseconds within a frame is far steadier than comparing FPS on a
 * laptop whose clocks swing with temperature and power limits.
 */
final class GpuTimer implements AutoCloseable {
	/** Stamp points, in frame order. A stage's time is from the previous stamp to its own. */
	static final int WORLD_START = 0, WORLD_END = 1, MOTION_END = 2, DECIDE_END = 3, TEMPORAL_END = 4, UPSCALE_END = 5;
	static final String[] STAGES = {"", "world", "motion", "decide", "temporal", "sharpen"};
	private static final int STAMPS = 6;
	private static final int SLOTS = 4; // frames in flight before a slot is reused
	private static final double SMOOTHING = 0.05;

	private GpuQueryPool pool;
	private final boolean[] pending = new boolean[SLOTS];
	private final boolean[][] written = new boolean[SLOTS][STAMPS];
	private int slot;
	private final double[] stageMs = new double[STAMPS];
	private double worldMs = Double.NaN;
	private double upscaleMs = Double.NaN;
	/** Raw per-frame samples while a benchmark window is open; null otherwise. */
	private List<double[]> recording;

	GpuTimer() {
		java.util.Arrays.fill(stageMs, Double.NaN);
	}

	void stamp(int which) {
		if (pool == null) {
			pool = RenderSystem.getDevice().createTimestampQueryPool(SLOTS * STAMPS);
		}
		if (which == WORLD_START) {
			slot = (slot + 1) % SLOTS;
			collect(slot);
			java.util.Arrays.fill(written[slot], false);
		}
		written[slot][which] = true;
		RenderSystem.getDevice().createCommandEncoder().writeTimestamp(pool, slot * STAMPS + which);
		if (which == WORLD_END) {
			pending[slot] = true;
		}
	}

	private void collect(int s) {
		if (!pending[s]) {
			return;
		}
		pending[s] = false;
		OptionalLong[] t = pool.getValues(s * STAMPS, STAMPS);
		if (t[WORLD_START].isEmpty() || t[WORLD_END].isEmpty()) {
			return; // not ready yet; drop this sample rather than stall
		}
		double period = RenderSystem.getDevice().getDeviceInfo().timestampPeriod() / 1e6;
		double world = (t[WORLD_END].getAsLong() - t[WORLD_START].getAsLong()) * period;
		worldMs = smooth(worldMs, world);
		double upscale = Double.NaN;
		if (written[s][UPSCALE_END] && t[UPSCALE_END].isPresent()) {
			upscale = (t[UPSCALE_END].getAsLong() - t[WORLD_END].getAsLong()) * period;
			upscaleMs = smooth(upscaleMs, upscale);
			long previous = t[WORLD_END].getAsLong();
			for (int i = MOTION_END; i <= UPSCALE_END; i++) {
				if (written[s][i] && t[i].isPresent()) {
					stageMs[i] = smooth(stageMs[i], (t[i].getAsLong() - previous) * period);
					previous = t[i].getAsLong();
				}
			}
		}
		if (recording != null) {
			recording.add(new double[] {world, upscale});
		}
	}

	private static double smooth(double old, double sample) {
		return Double.isNaN(old) ? sample : old + (sample - old) * SMOOTHING;
	}

	void startRecording() {
		recording = new ArrayList<>();
	}

	/** @return per-frame {world ms, upscale ms (NaN when off)} since startRecording */
	List<double[]> stopRecording() {
		List<double[]> out = recording == null ? List.of() : recording;
		recording = null;
		return out;
	}

	/** Forget averages, e.g. after changing quality, so old numbers don't blend into new ones. */
	void reset() {
		java.util.Arrays.fill(pending, false); // frames still in flight belong to the old mode
		java.util.Arrays.fill(stageMs, Double.NaN);
		worldMs = Double.NaN;
		upscaleMs = Double.NaN;
	}

	double worldMs() {
		return worldMs;
	}

	/** NaN while upscaling is off. */
	double upscaleMs() {
		return upscaleMs;
	}

	/** Per-pass breakdown, e.g. "motion 0.05, decide 0.10, temporal 0.20, sharpen 0.06". */
	String stages() {
		StringBuilder b = new StringBuilder();
		for (int i = MOTION_END; i <= UPSCALE_END; i++) {
			if (!Double.isNaN(stageMs[i])) {
				b.append(b.isEmpty() ? "" : ", ").append(STAGES[i]).append(String.format(" %.2f", stageMs[i]));
			}
		}
		return b.toString();
	}

	@Override
	public void close() {
		if (pool != null) {
			pool.close();
			pool = null;
		}
	}
}
