package dev.minesr.test;

import dev.minesr.client.MineSRClient;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.client.InactivityFpsLimit;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * A recording bot: travels to many places, varies time, weather and mobs, and moves around
 * (walk, sprint-jump, fly, look around, fast turns, strafe) while capture runs.
 * Only runs with MINESR_TEST=dataset. Output: minesr/captures/ (survives test-folder cleanup).
 */
public class MineSRDatasetTest implements FabricClientGameTest {
	private static final Logger LOG = LoggerFactory.getLogger("minesr-test");
	private static final int MOVE_TICKS = 100; // 5 seconds per movement segment

	private record Scene(String dimension, int x, int z, int time, boolean rain, List<String> moves, List<String> mobs) {
	}

	private static final List<String> FARM = List.of("cow", "sheep", "pig", "chicken", "horse", "villager");
	private static final List<String> HOSTILE = List.of("zombie", "skeleton", "creeper", "spider");

	private static final List<Scene> SCENES = List.of(
			new Scene("overworld", 0, 0, 1000, false, List.of("walk", "look", "sprint"), FARM),
			new Scene("overworld", 3000, 0, 6000, false, List.of("fly", "flick", "walk"), FARM),
			new Scene("overworld", -3000, 2000, 12500, false, List.of("sprint", "strafe", "look"), FARM),
			new Scene("overworld", 6000, -4000, 18000, false, List.of("walk", "flick", "fly"), HOSTILE),
			new Scene("overworld", -8000, -6000, 6000, true, List.of("look", "walk", "sprint"), FARM),
			new Scene("overworld", 12000, 9000, 3000, false, List.of("fly", "look", "strafe"), FARM),
			new Scene("overworld", 20000, -15000, 13500, true, List.of("sprint", "flick", "walk"), HOSTILE),
			new Scene("overworld", -25000, 18000, 9000, false, List.of("strafe", "fly", "look"), FARM),
			new Scene("the_nether", 0, 0, 6000, false, List.of("walk", "look", "flick"), List.of("piglin", "hoglin", "zombified_piglin")),
			new Scene("the_nether", 1500, 1200, 6000, false, List.of("fly", "strafe", "sprint"), List.of("strider", "magma_cube", "piglin")),
			new Scene("the_nether", -900, 700, 6000, false, List.of("look", "walk", "fly"), List.of("ghast", "blaze", "zombified_piglin")),
			// End: viewpoints around the main island, facing its center (out here the void is all black).
			new Scene("the_end", 45, 0, 6000, false, List.of("look", "fly", "walk"), List.of("enderman", "enderman", "endermite")),
			new Scene("the_end", 0, 55, 6000, false, List.of("fly", "look", "flick"), List.of("enderman", "enderman")),
			new Scene("the_end", -50, -50, 6000, false, List.of("strafe", "fly", "look"), List.of("enderman", "enderman")));

	@Override
	public void runTest(ClientGameTestContext context) {
		if (!"dataset".equals(System.getenv("MINESR_TEST"))) {
			return; // long-running; only on request
		}
		// MINESR_SEED picks another world (and shifts every destination), for more varied data.
		String seed = System.getenv().getOrDefault("MINESR_SEED", "minesr-dataset");
		int shift = seed.equals("minesr-dataset") ? 0 : Math.floorMod(seed.hashCode(), 50) * 1000 + 40000;
		Random random = new Random(seed.hashCode());
		try (TestSingleplayerContext world = context.worldBuilder()
				.setUseConsistentSettings(false)
				.adjustSettings(settings -> settings.setSeed(seed))
				.create()) {
			world.getServer().runCommand("gamemode creative @a");
			world.getServer().runCommand("gamerule advance_time false");
			context.runOnClient(mc -> {
				mc.options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
				mc.options.renderDistance().set(8);
				mc.options.autoJump().set(true); // climb one-block steps instead of walking into them
				mc.getWindow().setWindowed(1280, 720); // 16:9 like the capture target
				if (!mc.gui.hud.isHidden()) {
					mc.gui.hud.toggle();
				}
				var config = MineSRClient.config();
				config.captureBurstFrames = 32;
				config.captureBurstIntervalSeconds = 3.0f;
			});

			String onlyDimension = System.getenv("MINESR_DIMENSION"); // e.g. "the_end" or "overworld,the_nether"
			for (int i = 0; i < SCENES.size(); i++) {
				Scene base = SCENES.get(i);
				// The End island is only at the center; elsewhere, move every destination by the shift.
				boolean end = base.dimension().equals("the_end");
				int sx = end ? 0 : (base.dimension().equals("the_nether") ? shift / 8 : shift);
				Scene scene = new Scene(base.dimension(), base.x() + sx, base.z() - sx, base.time(), base.rain(), base.moves(), base.mobs());
				if (onlyDimension != null && !onlyDimension.isBlank() && !List.of(onlyDimension.split(",")).contains(scene.dimension())) {
					continue;
				}
				String onlyScenes = System.getenv("MINESR_SCENES"); // 1-based, e.g. "10,12,13,14"
				if (onlyScenes != null && !onlyScenes.isBlank() && !List.of(onlyScenes.split(",")).contains(String.valueOf(i + 1))) {
					continue;
				}
				int fps = i % 2 == 0 ? 60 : 90;
				LOG.info("Dataset scene {}/{}: {} at {}, {} | time {} {} | {} fps | {}", i + 1, SCENES.size(), scene.dimension(),
						scene.x(), scene.z(), scene.time(), scene.rain() ? "rain" : "clear", fps, scene.moves());
				travel(context, world, scene);
				world.getServer().runCommand("time set " + scene.time());
				world.getServer().runCommand("weather " + (scene.rain() ? "rain" : "clear"));
				for (int m = 0; m < 4; m++) {
					String mob = scene.mobs().get(random.nextInt(scene.mobs().size()));
					world.getServer().runCommand(String.format("execute at @p run summon minecraft:%s ~%d ~1 ~%d",
							mob, random.nextInt(13) - 6, random.nextInt(9) + 4));
				}
				context.runOnClient(mc -> mc.options.framerateLimit().set(fps));
				context.waitTicks(20);

				context.runOnClient(mc -> MineSRClient.upscaler().capture().start(outputDir(mc.gameDirectory.toPath())));
				// MINESR_MOVES=calm replaces each scene's moves with standing still and slow motion:
				// the moments where flicker is most visible.
				boolean calm = "calm".equals(System.getenv("MINESR_MOVES"));
				for (String move : calm ? List.of("still", "slowpan", "slowwalk") : scene.moves()) {
					move(context, world, move, random);
				}
				context.runOnClient(mc -> MineSRClient.upscaler().capture().stop());
				boolean written = context.computeOnClient(mc -> {
					try {
						return MineSRClient.upscaler().capture().awaitWriter(120);
					} catch (InterruptedException e) {
						return false;
					}
				});
				if (!written) {
					LOG.warn("  writer still busy after 2 minutes");
				}
				String done = context.computeOnClient(mc -> MineSRClient.upscaler().capture().describe());
				LOG.info("  -> {}", done);
			}
		}
	}

	/**
	 * minesr/captures/, outside the test run folder: Loom deletes that folder before every
	 * test run, which would throw away everything recorded so far.
	 */
	private static java.nio.file.Path outputDir(java.nio.file.Path runDir) {
		return runDir.toAbsolutePath().getParent().getParent().getParent().resolve("captures");
	}

	private static void travel(ClientGameTestContext context, TestSingleplayerContext world, Scene scene) {
		String in = "execute in minecraft:" + scene.dimension() + " run ";
		setFlying(context, world, false);
		switch (scene.dimension()) {
			case "the_nether" -> world.getServer().runCommand(in + String.format("spreadplayers %d %d 0 32 under 100 false @a", scene.x(), scene.z()));
			case "the_end" -> {
				// Hover above the island's edge, looking toward the middle (pillars, portal, dragon).
				float yaw = (float) Math.toDegrees(Math.atan2(scene.x(), -scene.z()));
				setFlying(context, world, true);
				// Locale.ROOT: on a Russian-locale system %.1f writes "90,0", which the command rejects.
				world.getServer().runCommand(in + String.format(java.util.Locale.ROOT, "tp @a %d 75 %d %.1f 15", scene.x(), scene.z(), yaw));
			}
			default -> world.getServer().runCommand(in + String.format("spreadplayers %d %d 0 32 false @a", scene.x(), scene.z()));
		}
		try {
			world.getConnection().waitForChunksRender(400);
		} catch (AssertionError timeout) {
			// Far chunks can keep loading; enough is on screen to move around.
		}
		context.waitTicks(scene.dimension().equals("overworld") ? 60 : 200); // other dimensions load from scratch
	}

	private static void move(ClientGameTestContext context, TestSingleplayerContext world, String move, Random random) {
		var input = context.getInput();
		Function<Options, KeyMapping> up = o -> o.keyUp, jump = o -> o.keyJump, sprint = o -> o.keySprint;
		Function<Options, KeyMapping> left = o -> o.keyLeft, right = o -> o.keyRight;
		float turn = (random.nextFloat() - 0.5f) * 3.0f;
		setFlying(context, world, move.equals("fly"));
		if (move.equals("fly")) {
			// Climb first, so the flight goes over the terrain rather than into it.
			world.getServer().runCommand("execute as @p at @s run tp @s ~ ~12 ~");
			context.waitTicks(5);
		}
		switch (move) {
			case "walk" -> input.holdKey(up);
			case "sprint" -> {
				input.holdKey(up);
				input.holdKey(sprint);
				input.holdKey(jump);
			}
			case "fly" -> {
				input.holdKey(up);
				input.holdKey(sprint);
			}
			case "strafe" -> input.holdKey(left);
			case "slowwalk" -> input.holdKey(o -> o.keyShift); // sneaking: slow, steady movement
			case "still", "slowpan" -> {
			}
			default -> {
			}
		}
		int[] wallTurn = {0};
		for (int t = 0; t < MOVE_TICKS; t++) {
			int tick = t;
			if (move.equals("slowwalk") && t == 0) {
				input.holdKey(up);
			}
			if (move.equals("walk") && t % 25 == 10) {
				input.pressKey(jump);
			}
			if (move.equals("strafe") && t == MOVE_TICKS / 2) {
				input.releaseKey(left);
				input.holdKey(right);
			}
			boolean inLava = context.computeOnClient(mc -> mc.player.isInLava());
			if (inLava) { // frames of solid lava teach nothing: get out and keep flying
				world.getServer().runCommand("execute as @p at @s run tp @s ~ ~10 ~");
				setFlying(context, world, true);
			}
			context.runOnClient(mc -> {
				var p = mc.player;
				if (p.horizontalCollision && !move.equals("look") && wallTurn[0] == 0) {
					wallTurn[0] = 5; // bumped into something: turn away from the wall over 5 ticks
				}
				if (wallTurn[0] > 0) {
					wallTurn[0]--;
					p.setYRot(p.getYRot() + 22.0f); // gradual, so the recording has no camera cut
				}
				switch (move) {
					case "look" -> {
						p.setYRot(p.getYRot() + 2.5f * (float) Math.sin(tick / 20.0));
						p.setXRot((float) (15 * Math.sin(tick / 13.0)));
					}
					case "still" -> {
						// nothing: only the jitter changes between frames
					}
					case "slowpan" -> p.setYRot(p.getYRot() + 0.3f); // 6 degrees per second
					case "flick" -> {
						if (tick % 20 < 3) {
							p.setYRot(p.getYRot() + 25.0f); // fast turns: the hardest case for history
						}
					}
					case "fly" -> {
						p.setYRot(p.getYRot() + turn);
						p.setXRot((float) (10 + 15 * Math.sin(tick / 18.0)));
					}
					default -> {
						p.setYRot(p.getYRot() + turn * (float) Math.sin(tick / 30.0));
						p.setXRot(8.0f);
					}
				}
			});
			context.waitTick();
		}
		input.releaseKey(up);
		input.releaseKey(sprint);
		input.releaseKey(jump);
		input.releaseKey(left);
		input.releaseKey(right);
		input.releaseKey(o -> o.keyShift);
	}

	private static void setFlying(ClientGameTestContext context, TestSingleplayerContext world, boolean flying) {
		world.getServer().runOnServer(server -> {
			var player = server.getPlayerList().getPlayers().getFirst();
			player.getAbilities().flying = flying;
			player.onUpdateAbilities();
		});
		context.runOnClient(mc -> mc.player.getAbilities().flying = flying);
	}
}
