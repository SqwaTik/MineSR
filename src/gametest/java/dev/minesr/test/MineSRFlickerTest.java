package dev.minesr.test;

import dev.minesr.Quality;
import dev.minesr.client.MineSRClient;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.fabricmc.fabric.api.client.gametest.v1.screenshot.TestScreenshotOptions;
import net.minecraft.client.InactivityFpsLimit;

import java.nio.file.Path;

/**
 * Flicker hunt: the camera stands perfectly still, so anything that changes between consecutive
 * frames is either legitimate animation (water, clouds; vanilla shows how much) or a bug.
 * Saves 12 consecutive frames per mode to build/run/clientGameTest/flicker/<mode>/.
 * Analyze with training/analyze_flicker.py. Run with MINESR_TEST=flicker.
 */
public class MineSRFlickerTest implements FabricClientGameTest {
	private record Mode(String name, boolean enabled, Quality quality, boolean neural, boolean grid, float sharpness) {
		Mode(String name, boolean enabled, Quality quality, boolean neural, boolean grid) {
			this(name, enabled, quality, neural, grid, 0.5f);
		}
	}

	private static final Mode[] MODES = {
			new Mode("vanilla", false, Quality.NATIVE, false, true),
			new Mode("quality-hand-free", true, Quality.QUALITY, false, false),
			new Mode("quality-hand-grid", true, Quality.QUALITY, false, true),
			new Mode("quality-neural-free", true, Quality.QUALITY, true, false),
			new Mode("quality-neural-grid", true, Quality.QUALITY, true, true),
			new Mode("performance-hand-grid", true, Quality.PERFORMANCE, false, true),
			new Mode("performance-neural-grid", true, Quality.PERFORMANCE, true, true),
			new Mode("quality-hand-grid-nosharpen", true, Quality.QUALITY, false, true, 0.0f),
			new Mode("quality-neural-grid-nosharpen", true, Quality.QUALITY, true, true, 0.0f),
	};

	@Override
	public void runTest(ClientGameTestContext context) {
		if (!"flicker".equals(System.getenv("MINESR_TEST"))) {
			return;
		}
		try (TestSingleplayerContext world = context.worldBuilder()
				.setUseConsistentSettings(false)
				.adjustSettings(settings -> settings.setSeed("minesr"))
				.create()) {
			world.getServer().runCommand("gamemode creative @a");
			world.getServer().runCommand("gamerule advance_time false");
			world.getServer().runCommand("gamerule spawn_mobs false");
			world.getServer().runOnServer(server -> {
				var player = server.getPlayerList().getPlayers().getFirst();
				player.getAbilities().flying = true;
				player.onUpdateAbilities();
			});
			world.getServer().runCommand("time set 6000");
			world.getServer().runCommand("weather clear");
			world.getServer().runCommand("tp @a 0 110 0 35 30");
			context.runOnClient(mc -> {
				mc.options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
				mc.options.framerateLimit().set(60);
				mc.getWindow().setWindowed(1280, 720);
				if (!mc.gui.hud.isHidden()) {
					mc.gui.hud.toggle();
				}
			});
			world.getConnection().waitForChunksRender();
			context.waitTicks(100);

			for (Mode mode : MODES) {
				context.runOnClient(mc -> {
					MineSRClient.config().enabled = mode.enabled();
					MineSRClient.config().quality = mode.quality().id();
					MineSRClient.config().neural = mode.neural();
					MineSRClient.config().gridJitter = mode.grid();
					MineSRClient.config().sharpness = mode.sharpness();
					if (!mode.enabled()) {
						MineSRClient.upscaler().release();
					}
					MineSRClient.upscaler().resetHistory();
				});
				context.waitTicks(60); // history fully settled
				Path dir = context.computeOnClient(mc -> mc.gameDirectory.toPath().resolve("flicker").resolve(mode.name()));
				for (int i = 0; i < 12; i++) {
					context.takeScreenshot(TestScreenshotOptions.of(String.format("%02d", i))
							.withDestinationDir(dir).disableCounterPrefix());
				}
			}
		}
	}
}
