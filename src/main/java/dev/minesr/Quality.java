package dev.minesr;

import java.util.Locale;

/** Render-resolution presets, named and scaled like FSR's quality modes. */
public enum Quality {
	NATIVE(1.0f),       // full resolution, temporal anti-aliasing only
	QUALITY(1.0f / 1.5f),
	BALANCED(1.0f / 1.7f),
	PERFORMANCE(0.5f);

	public final float scale;

	Quality(float scale) {
		this.scale = scale;
	}

	public String id() {
		return name().toLowerCase(Locale.ROOT);
	}

	/** Low-res size for an output size, never below 1. */
	public static int scaled(int outSize, float scale) {
		return Math.max(1, Math.round(outSize * scale));
	}

	public static Quality byId(String id) {
		for (Quality q : values()) {
			if (q.id().equals(id)) {
				return q;
			}
		}
		return null;
	}
}
