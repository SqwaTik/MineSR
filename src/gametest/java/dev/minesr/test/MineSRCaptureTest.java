package dev.minesr.test;

import dev.minesr.client.MineSRClient;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.client.InactivityFpsLimit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Records one burst of training data while flying forward and turning, then stops.
 * Check the output with {@code python training/check_capture.py <session dir>}.
 * Set MINESR_TEST=capture to run only this test.
 */
public class MineSRCaptureTest implements FabricClientGameTest {
	private static final Logger LOG = LoggerFactory.getLogger("minesr-test");

	@Override
	public void runTest(ClientGameTestContext context) {
		if (!MineSRScreenshotTest.selected("capture")) {
			return;
		}
		try (TestSingleplayerContext world = context.worldBuilder()
				.setUseConsistentSettings(false)
				.adjustSettings(settings -> settings.setSeed("minesr"))
				.create()) {
			world.getServer().runCommand("gamemode creative @a");
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
				mc.options.framerateLimit().set(60); // spread a burst over real motion
				if (!mc.gui.hud.isHidden()) {
					mc.gui.hud.toggle();
				}
				var config = MineSRClient.config();
				config.captureBurstFrames = 24;
				config.captureBurstIntervalSeconds = 3600; // just one burst
			});
			world.getConnection().waitForChunksRender();
			context.waitTicks(40);

			context.runOnClient(mc -> MineSRClient.upscaler().capture().start(mc.gameDirectory.toPath().resolve("minesr-captures")));
			// Burst starts 2 s after start; move for 4 s so it lands mid-motion.
			for (int i = 0; i < 80; i++) {
				context.runOnClient(mc -> {
					var p = mc.player;
					p.setYRot(p.getYRot() + 1.5f);
					double yaw = Math.toRadians(p.getYRot());
					p.setDeltaMovement(-Math.sin(yaw) * 0.4, 0, Math.cos(yaw) * 0.4);
				});
				context.waitTick();
			}
			context.runOnClient(mc -> MineSRClient.upscaler().capture().stop());
			boolean done = context.computeOnClient(mc -> {
				try {
					return MineSRClient.upscaler().capture().awaitWriter(60);
				} catch (InterruptedException e) {
					return false;
				}
			});
			String status = context.computeOnClient(mc -> MineSRClient.upscaler().capture().describe());
			LOG.info("MineSR capture test: writer finished={} {} -> {}", done, status,
					context.computeOnClient(mc -> MineSRClient.upscaler().capture().session().toAbsolutePath()));
			if (MineSRClient.upscaler().failure() != null) {
				throw new AssertionError("MineSR failed: " + MineSRClient.upscaler().failure());
			}
		}
	}
}
