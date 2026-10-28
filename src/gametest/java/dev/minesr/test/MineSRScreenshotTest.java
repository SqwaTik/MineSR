package dev.minesr.test;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import dev.minesr.Quality;
import dev.minesr.client.MineSRClient;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.client.InactivityFpsLimit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Opens a real world, screenshots each mode from the same spot, then benchmarks.
 * Run with {@code gradlew runClientGameTest}. Output goes to {@code build/run/clientGameTest}:
 * {@code screenshots/} and {@code minesr-benchmark.json}.
 */
public class MineSRScreenshotTest implements FabricClientGameTest {
	private static final Logger LOG = LoggerFactory.getLogger("minesr-test");

	/** A benchmark mode: upscaler on/off, preset, and which version of the upscale pass. */
	private record Mode(String name, boolean enabled, Quality quality, int pass, boolean neural) {
	}

	private static final Mode[] MODES = {
			new Mode("off (vanilla)", false, Quality.NATIVE, 2, false),
			new Mode("quality, hand-tuned", true, Quality.QUALITY, 2, false),
			new Mode("quality, neural", true, Quality.QUALITY, 2, true),
			new Mode("balanced, neural", true, Quality.BALANCED, 2, true),
			new Mode("performance, hand-tuned", true, Quality.PERFORMANCE, 2, false),
			new Mode("performance, neural", true, Quality.PERFORMANCE, 2, true),
	};

	/** MINESR_TEST picks which test runs (e.g. "capture", "bench"); unset runs everything. */
	static boolean selected(String name) {
		String only = System.getenv("MINESR_TEST");
		return only == null || only.isBlank() ? !name.equals("dataset") && !name.equals("flicker") : only.equals(name);
	}

	@Override
	public void runTest(ClientGameTestContext context) {
		if (!selected("bench")) {
			return;
		}
		try (TestSingleplayerContext world = context.worldBuilder()
				.setUseConsistentSettings(false)
				.adjustSettings(settings -> settings.setSeed("minesr"))
				.create()) {
			world.getServer().runCommand("gamemode creative @a");
			world.getServer().runOnServer(server -> {
				var player = server.getPlayerList().getPlayers().getFirst();
				player.getAbilities().flying = true; // hover in place instead of falling
				player.onUpdateAbilities();
				server.getPlayerList().setViewDistance(12); // a heavier scene, closer to real play
			});
			world.getServer().runCommand("time set 6000");
			world.getServer().runCommand("weather clear");
			world.getServer().runCommand("tp @a 0 110 0 35 30");
			context.runOnClient(mc -> {
				mc.options.renderDistance().set(12);
				mc.options.enableVsync().set(false);
				mc.options.framerateLimit().set(260); // 260 = unlimited
				// The test sends no input; without this the game decides we're AFK and caps at 30 FPS.
				mc.options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
				mc.getWindow().setWindowed(1600, 900);
				if (!mc.gui.hud.isHidden()) {
					mc.gui.hud.toggle();
				}
			});
			context.waitTicks(1200); // a minute to load render distance 12 (the all-chunks check never settles at this distance)
			context.waitTicks(100);

			shoot(context, "1-off", MODES[0]);
			shoot(context, "2-quality-hand", MODES[1]);
			shoot(context, "3-quality-neural", MODES[2]);
			shoot(context, "4-performance-hand", MODES[4]);
			shoot(context, "5-performance-neural", MODES[5]);

			// Turning: history has to follow the camera without smearing.
			configure(context, MODES[2]);
			context.waitTicks(20);
			for (int i = 0; i < 15; i++) {
				context.runOnClient(mc -> mc.player.setYRot(mc.player.getYRot() + 4.0f));
				context.waitTick();
			}
			context.takeScreenshot("6-quality-turning");
			configure(context, MODES[0]);
			context.waitTicks(5);
			context.takeScreenshot("7-off-turned");
			context.waitTicks(100);

			JsonObject results = new JsonObject();
			results.add("default graphics", benchmark(context, "default graphics"));
			// Heavier on the GPU: improved transparency adds full-screen passes that scale with resolution.
			context.runOnClient(mc -> mc.options.improvedTransparency().set(true));
			context.waitTicks(200);
			results.add("improved transparency", benchmark(context, "improved transparency"));
			Path file = context.computeOnClient(mc -> mc.gameDirectory.toPath().resolve("minesr-benchmark.json"));
			try {
				Files.writeString(file, new GsonBuilder().setPrettyPrinting().serializeNulls().create().toJson(results));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}

			if (MineSRClient.upscaler().failure() != null) {
				throw new AssertionError("MineSR failed: " + MineSRClient.upscaler().failure());
			}
		}
	}

	/**
	 * Interleaved: every cycle visits every mode for about a second. Slow swings in GPU clock
	 * speed (heat, power limits) then hit all modes alike instead of whichever ran last.
	 * Frame rate comes from a frame counter over a measured time window.
	 */
	private static JsonObject benchmark(ClientGameTestContext context, String label) {
		int cycles = 10;
		List<List<Double>> fps = new ArrayList<>();
		List<List<Double>> worldMs = new ArrayList<>();
		List<List<Double>> upscaleMs = new ArrayList<>();
		for (int i = 0; i < MODES.length; i++) {
			fps.add(new ArrayList<>());
			worldMs.add(new ArrayList<>());
			upscaleMs.add(new ArrayList<>());
		}
		for (int cycle = 0; cycle < cycles; cycle++) {
			for (int i = 0; i < MODES.length; i++) {
				configure(context, MODES[i]);
				context.waitTicks(10); // let history and buffers settle
				long[] start = context.computeOnClient(mc -> new long[] {MineSRClient.upscaler().frameCount(), System.nanoTime()});
				context.runOnClient(mc -> MineSRClient.upscaler().startGpuRecording());
				context.waitTicks(20);
				long[] end = context.computeOnClient(mc -> new long[] {MineSRClient.upscaler().frameCount(), System.nanoTime()});
				List<double[]> gpu = context.computeOnClient(mc -> MineSRClient.upscaler().stopGpuRecording());
				fps.get(i).add((end[0] - start[0]) / ((end[1] - start[1]) / 1e9));
				if (cycle == cycles - 1) {
					String detail = context.computeOnClient(mc -> MineSRClient.upscaler().describe());
					LOG.info("  [{}] {}", MODES[i].name(), detail);
				}
				for (double[] sample : gpu) {
					worldMs.get(i).add(sample[0]);
					if (!Double.isNaN(sample[1])) {
						upscaleMs.get(i).add(sample[1]);
					}
				}
			}
		}

		String resolution = context.computeOnClient(mc -> mc.getWindow().getWidth() + "x" + mc.getWindow().getHeight());
		JsonObject out = new JsonObject();
		out.addProperty("resolution", resolution);
		out.addProperty("renderDistance", 12);
		out.addProperty("cycles", cycles);
		JsonArray modes = new JsonArray();
		StringBuilder table = new StringBuilder(String.format(
				"MineSR benchmark [%s], %s, render distance 12, %d interleaved cycles (medians):%n", label, resolution, cycles));
		double vanillaFps = median(fps.get(0)); // MODES[0] is the mod switched off
		table.append(String.format("  %-26s %12s %10s %8s %10s %11s%n",
				"mode", "FPS without", "FPS with", "change", "world ms", "upscale ms"));
		for (int i = 0; i < MODES.length; i++) {
			JsonObject m = new JsonObject();
			m.addProperty("name", MODES[i].name());
			m.addProperty("fps", median(fps.get(i)));
			m.addProperty("fpsP25", percentile(fps.get(i), 0.25));
			m.addProperty("fpsP75", percentile(fps.get(i), 0.75));
			m.addProperty("worldMs", median(worldMs.get(i)));
			m.addProperty("upscaleMs", upscaleMs.get(i).isEmpty() ? null : median(upscaleMs.get(i)));
			m.addProperty("gpuFrames", worldMs.get(i).size());
			JsonArray perCycle = new JsonArray();
			fps.get(i).forEach(perCycle::add);
			m.add("fpsPerCycle", perCycle);
			modes.add(m);
			double withFps = median(fps.get(i));
			table.append(String.format("  %-26s %12.0f %10.0f %+7.0f%% %10.2f %11s%n", MODES[i].name(), vanillaFps, withFps,
					(withFps / vanillaFps - 1) * 100, median(worldMs.get(i)),
					upscaleMs.get(i).isEmpty() ? "-" : String.format("%.2f", median(upscaleMs.get(i)))));
		}
		out.add("modes", modes);
		LOG.info(table.toString());
		return out;
	}

	private static double median(List<Double> values) {
		return percentile(values, 0.5);
	}

	private static double percentile(List<Double> values, double p) {
		if (values.isEmpty()) {
			return Double.NaN;
		}
		double[] sorted = values.stream().mapToDouble(Double::doubleValue).toArray();
		Arrays.sort(sorted);
		return sorted[(int) Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1)))];
	}

	private static void shoot(ClientGameTestContext context, String name, Mode mode) {
		configure(context, mode);
		context.waitTicks(60); // let history settle
		context.takeScreenshot(name);
	}

	private static void configure(ClientGameTestContext context, Mode mode) {
		context.runOnClient(mc -> {
			MineSRClient.config().enabled = mode.enabled();
			MineSRClient.config().quality = mode.quality().id();
			MineSRClient.config().passVersion = mode.pass();
			MineSRClient.config().neural = mode.neural();
			if (!mode.enabled()) {
				MineSRClient.upscaler().release();
			}
			MineSRClient.upscaler().resetHistory();
		});
	}
}
