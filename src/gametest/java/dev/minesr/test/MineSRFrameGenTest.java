package dev.minesr.test;

import dev.minesr.Quality;
import dev.minesr.client.MineSRClient;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.client.InactivityFpsLimit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Frame generation check, with the HUD visible: rendered vs shown frames per second, how evenly
 * the shown frames are spaced, and a snapshot (previous / generated / current) while turning.
 * Run with MINESR_TEST=framegen. Snapshots: build/run/clientGameTest/framegen/<mode>/.
 */
public class MineSRFrameGenTest implements FabricClientGameTest {
	private static final Logger LOG = LoggerFactory.getLogger("minesr-test");

	private record Mode(String name, boolean upscale, Quality quality, boolean frameGen) {
	}

	private static final Mode[] MODES = {
			new Mode("vanilla", false, Quality.NATIVE, false),
			new Mode("framegen-only", false, Quality.NATIVE, true),
			new Mode("quality", true, Quality.QUALITY, false),
			new Mode("quality+framegen", true, Quality.QUALITY, true),
	};

	@Override
	public void runTest(ClientGameTestContext context) {
		if (!"framegen".equals(System.getenv("MINESR_TEST"))) {
			return;
		}
		try (TestSingleplayerContext world = context.worldBuilder()
				.setUseConsistentSettings(false)
				.adjustSettings(settings -> settings.setSeed("minesr"))
				.create()) {
			world.getServer().runCommand("gamemode creative @a");
			world.getServer().runCommand("gamerule advance_time false");
			world.getServer().runOnServer(server -> {
				var player = server.getPlayerList().getPlayers().getFirst();
				player.getAbilities().flying = true;
				player.onUpdateAbilities();
			});
			world.getServer().runCommand("give @a minecraft:diamond_sword");
			world.getServer().runCommand("time set 6000");
			world.getServer().runCommand("weather clear");
			world.getServer().runCommand("tp @a 0 110 0 35 20");
			context.runOnClient(mc -> {
				mc.options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
				mc.options.enableVsync().set(false);
				mc.options.framerateLimit().set(260);
				mc.getWindow().setWindowed(1280, 720);
			});
			world.getConnection().waitForChunksRender();
			context.waitTicks(60);

			StringBuilder table = new StringBuilder("MineSR frame generation (turning 3 degrees per tick, HUD visible):\n");
			table.append(String.format("  %-18s %13s %11s %22s%n", "mode", "rendered FPS", "shown FPS", "shown-frame spacing"));
			for (Mode mode : MODES) {
				context.runOnClient(mc -> {
					var c = MineSRClient.config();
					c.enabled = mode.upscale();
					c.quality = mode.quality().id();
					c.frameGeneration = mode.frameGen();
					if (!mode.upscale()) {
						MineSRClient.upscaler().release();
					}
					if (!mode.frameGen()) {
						MineSRClient.frameGen().release();
					}
					MineSRClient.upscaler().resetHistory();
				});
				context.waitTicks(20);
				long[] start = context.computeOnClient(mc -> new long[] {
						MineSRClient.upscaler().frameCount(), MineSRClient.frameGen().presentCount(), System.nanoTime()});
				context.runOnClient(mc -> MineSRClient.frameGen().startPresentTiming());
				for (int t = 0; t < 60; t++) {
					context.runOnClient(mc -> mc.player.setYRot(mc.player.getYRot() + 3.0f));
					if (t == 40 && mode.frameGen()) {
						Path dir = context.computeOnClient(mc -> mc.gameDirectory.toPath().resolve("framegen").resolve(mode.name()));
						context.runOnClient(mc -> MineSRClient.frameGen().requestSnapshot(dir));
					}
					context.waitTick();
				}
				long[] end = context.computeOnClient(mc -> new long[] {
						MineSRClient.upscaler().frameCount(), MineSRClient.frameGen().presentCount(), System.nanoTime()});
				List<Long> times = context.computeOnClient(mc -> MineSRClient.frameGen().stopPresentTiming());
				double seconds = (end[2] - start[2]) / 1e9;
				double rendered = (end[0] - start[0]) / seconds;
				double shown = mode.frameGen() ? (end[1] - start[1]) / seconds : rendered;
				table.append(String.format("  %-18s %13.0f %11.0f %22s%n", mode.name(), rendered, shown, spacing(times)));
			}
			context.waitTicks(20); // let snapshot readbacks finish
			LOG.info(table.toString());
		}
	}

	/** Evenness of present intervals: 10th / 50th / 90th percentile, in ms. */
	private static String spacing(List<Long> times) {
		if (times.size() < 10) {
			return "-";
		}
		double[] gaps = new double[times.size() - 1];
		for (int i = 1; i < times.size(); i++) {
			gaps[i - 1] = (times.get(i) - times.get(i - 1)) / 1e6;
		}
		Arrays.sort(gaps);
		return String.format("%.1f / %.1f / %.1f ms", gaps[gaps.length / 10], gaps[gaps.length / 2], gaps[gaps.length * 9 / 10]);
	}
}
