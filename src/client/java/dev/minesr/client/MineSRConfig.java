package dev.minesr.client;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import dev.minesr.Quality;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;

/** {@code config/minesr.json}. Unknown or missing fields fall back to defaults. */
public final class MineSRConfig {
	private static final Logger LOG = LoggerFactory.getLogger("minesr");
	private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

	public boolean enabled = true;
	public String quality = Quality.QUALITY.id();
	/** 0 = off, 1 = strongest. */
	public float sharpness = 0.5f;
	/** Frames of history blended at most. Higher = smoother and more stable, slower to react. */
	public float maxHistory = 10.0f;
	/** Sharper textures at low resolution so the upscaler has detail to recover. */
	public boolean lodBias = true;
	/** Use the trained network for the upscale decisions; /minesr neural off for hand-tuned rules. */
	public boolean neural = true;
	/**
	 * Frame generation: an extra frame between every two rendered frames. Smoother motion,
	 * not faster controls (about half a frame of extra input delay). Off by default.
	 */
	public boolean frameGeneration = false;
	/** Jitter on the same fixed sub-pixel grid the network was trained on (instead of free positions). */
	public boolean gridJitter = true;
	/** 2 = current upscale pass, 1 = the first version (kept for benchmarking). */
	public int passVersion = 2;

	// Training-data capture (/minesr capture start)
	/** World render size while capturing; rounded down to a multiple of 12. Target images are half this. */
	public int captureWidth = 1920;
	public int captureHeight = 1080;
	/** Consecutive frames per burst. Sequences teach the network how frames follow each other. */
	public int captureBurstFrames = 32;
	/** Pause between bursts, so a session covers many different places. */
	public float captureBurstIntervalSeconds = 10.0f;
	/** Stop saving once a session reaches this size. */
	public int captureMaxMegabytes = 4096;

	public Quality qualityPreset() {
		Quality q = Quality.byId(quality);
		return q != null ? q : Quality.QUALITY;
	}

	private static Path file() {
		return FabricLoader.getInstance().getConfigDir().resolve("minesr.json");
	}

	public static MineSRConfig load() {
		MineSRConfig config = new MineSRConfig();
		Path file = file();
		if (Files.exists(file)) {
			try (Reader reader = Files.newBufferedReader(file)) {
				MineSRConfig read = GSON.fromJson(reader, MineSRConfig.class);
				if (read != null) {
					config = read;
				}
			} catch (IOException | JsonParseException e) {
				LOG.warn("Could not read {}, using defaults: {}", file, e.getMessage());
			}
		}
		config.quality = config.qualityPreset().id();
		config.sharpness = Math.clamp(config.sharpness, 0.0f, 1.0f);
		config.maxHistory = Math.clamp(config.maxHistory, 2.0f, 32.0f);
		config.passVersion = config.passVersion == 1 ? 1 : 2;
		config.captureWidth = Math.max(240, Math.min(3840, config.captureWidth) / 12 * 12);
		config.captureHeight = Math.max(144, Math.min(2160, config.captureHeight) / 12 * 12);
		config.captureBurstFrames = Math.clamp(config.captureBurstFrames, 2, 600);
		config.captureBurstIntervalSeconds = Math.clamp(config.captureBurstIntervalSeconds, 0.0f, 3600.0f);
		config.captureMaxMegabytes = Math.clamp(config.captureMaxMegabytes, 10, 1_000_000);
		config.save();
		return config;
	}

	public void save() {
		try (Writer writer = Files.newBufferedWriter(file())) {
			GSON.toJson(this, writer);
		} catch (IOException e) {
			LOG.warn("Could not save {}: {}", file(), e.getMessage());
		}
	}
}
