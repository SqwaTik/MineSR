package dev.minesr.client;

import com.mojang.brigadier.arguments.FloatArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import dev.minesr.Quality;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.network.chat.Component;

public final class MineSRClient implements ClientModInitializer {
	private static MineSRConfig config;
	private static Upscaler upscaler;
	private static FrameGen frameGen;
	private ClientLevel lastLevel;

	public static Upscaler upscaler() {
		return upscaler;
	}

	public static MineSRConfig config() {
		return config;
	}

	public static FrameGen frameGen() {
		return frameGen;
	}

	/** Frame generation pacing: called at a few points in each frame to show the waiting real frame on time. */
	public static void presentPendingFrameIfDue() {
		presentPendingFrameIfDue(false);
	}

	public static void presentPendingFrameIfDue(boolean lastChance) {
		if (frameGen != null) {
			frameGen.maybePresentPending(((dev.minesr.client.mixin.MinecraftAccessor) Minecraft.getInstance()).minesr$windowSurface(), lastChance);
		}
	}

	@Override
	public void onInitializeClient() {
		config = MineSRConfig.load();
		upscaler = new Upscaler(config);
		frameGen = new FrameGen(config, upscaler);
		loadNetwork();
		ClientTickEvents.END_CLIENT_TICK.register(this::onTick);
		ClientCommandRegistrationCallback.EVENT.register((dispatcher, context) -> {
			LiteralArgumentBuilder<FabricClientCommandSource> root = ClientCommands.literal("minesr").executes(this::status);
			root.then(ClientCommands.literal("on").executes(ctx -> setEnabled(ctx, true)));
			root.then(ClientCommands.literal("off").executes(ctx -> setEnabled(ctx, false)));
			for (Quality q : Quality.values()) {
				root.then(ClientCommands.literal(q.id()).executes(ctx -> setQuality(ctx, q)));
			}
			root.then(ClientCommands.literal("sharpness")
					.then(ClientCommands.argument("value", FloatArgumentType.floatArg(0.0f, 1.0f)).executes(ctx -> {
						config.sharpness = FloatArgumentType.getFloat(ctx, "value");
						config.save();
						return status(ctx);
					})));
			root.then(ClientCommands.literal("capture")
					.executes(this::captureStatus)
					.then(ClientCommands.literal("start").executes(ctx -> {
						upscaler.capture().start(ctx.getSource().getClient().gameDirectory.toPath().resolve("minesr-captures"));
						upscaler.resetHistory();
						ctx.getSource().sendFeedback(Component.literal("MineSR capture started. Play normally: walk, fly, turn. "
								+ "Saves " + config.captureBurstFrames + "-frame bursts every " + (int) config.captureBurstIntervalSeconds
								+ " s to minesr-captures/. /minesr capture stop when done."));
						return 1;
					}))
					.then(ClientCommands.literal("stop").executes(ctx -> {
						upscaler.capture().stop();
						upscaler.resetHistory();
						return captureStatus(ctx);
					})));
			root.then(ClientCommands.literal("neural")
					.then(ClientCommands.literal("on").executes(ctx -> setNeural(ctx, true)))
					.then(ClientCommands.literal("off").executes(ctx -> setNeural(ctx, false))));
			root.then(ClientCommands.literal("framegen")
					.then(ClientCommands.literal("on").executes(ctx -> setFrameGen(ctx, true)))
					.then(ClientCommands.literal("off").executes(ctx -> setFrameGen(ctx, false))));
			root.then(ClientCommands.literal("retry").executes(ctx -> {
				upscaler.retry();
				return status(ctx);
			}));
			dispatcher.register(root);
		});
	}

	private void onTick(Minecraft mc) {
		// New world or dimension: the old history shows a different place.
		if (mc.level != lastLevel) {
			lastLevel = mc.level;
			upscaler.resetHistory();
		}
	}

	private int setEnabled(CommandContext<FabricClientCommandSource> ctx, boolean on) {
		config.enabled = on;
		config.save();
		if (!on) {
			upscaler.release();
		}
		return status(ctx);
	}

	private int setQuality(CommandContext<FabricClientCommandSource> ctx, Quality quality) {
		config.quality = quality.id();
		config.enabled = true;
		config.save();
		upscaler.resetHistory();
		return status(ctx);
	}

	/** config/minesr/upscaler_net.json if you trained your own, otherwise the network shipped in the mod. */
	private static void loadNetwork() {
		java.nio.file.Path custom = net.fabricmc.loader.api.FabricLoader.getInstance().getConfigDir().resolve("minesr/upscaler_net.json");
		try {
			String text;
			String source;
			if (java.nio.file.Files.exists(custom)) {
				text = java.nio.file.Files.readString(custom);
				source = custom.toString();
			} else {
				try (var in = MineSRClient.class.getResourceAsStream("/assets/minesr/upscaler_net.json")) {
					if (in == null) {
						upscaler.setNetwork(null, "none bundled");
						return;
					}
					text = new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
					source = "bundled";
				}
			}
			upscaler.setNetwork(dev.minesr.NeuralWeights.pack(com.google.gson.JsonParser.parseString(text).getAsJsonObject()), source);
		} catch (Exception e) {
			org.slf4j.LoggerFactory.getLogger("minesr").warn("Could not load the upscaler network, using hand-tuned rules: {}", e.toString());
			upscaler.setNetwork(null, "failed: " + e.getMessage());
		}
	}

	private int setFrameGen(CommandContext<FabricClientCommandSource> ctx, boolean on) {
		config.frameGeneration = on;
		config.save();
		if (!on) {
			frameGen.release();
		}
		upscaler.resetHistory();
		return status(ctx);
	}

	private int setNeural(CommandContext<FabricClientCommandSource> ctx, boolean on) {
		config.neural = on;
		config.save();
		upscaler.resetHistory();
		if (on && !upscaler.neuralActive()) {
			ctx.getSource().sendFeedback(Component.literal("No trained network loaded (" + upscaler.networkSource() + "); using hand-tuned rules."));
		}
		return status(ctx);
	}

	private int captureStatus(CommandContext<FabricClientCommandSource> ctx) {
		ctx.getSource().sendFeedback(Component.literal("MineSR capture: " + upscaler.capture().describe()));
		return 1;
	}

	private int status(CommandContext<FabricClientCommandSource> ctx) {
		ctx.getSource().sendFeedback(Component.literal("MineSR " + upscaler.describe() + "; " + frameGen.describe()));
		return 1;
	}
}
