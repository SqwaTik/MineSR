package dev.minesr;

/**
 * Texture LOD bias for the low-res world pass, applied by editing shader text at load time.
 *
 * <p>Rendering at 67% makes the GPU pick blurrier mipmaps, so texture detail never reaches
 * the upscaler. Terrain samples with explicit derivatives; scaling them by the render scale
 * picks the mip level the full-resolution frame would have used (a -log2(scale) LOD bias).
 *
 * <p>Every edit is an exact-text match. If a resource pack changed the shader, the edit is
 * skipped and the game renders as before, just without the bias.
 */
public final class ShaderPatch {
	public static final String UNIFORM = "MineSRLodScale";

	private static final String GLOBALS_LAST_MEMBER = "int UseRgss;";
	private static final String[][] DERIVATIVES = {
			{"vec2 du = dFdx(uv);", "vec2 du = dFdx(uv) * " + UNIFORM + ";"},
			{"vec2 dv = dFdy(uv);", "vec2 dv = dFdy(uv) * " + UNIFORM + ";"},
	};

	private ShaderPatch() {
	}

	/** @param source a shader with its imports already inlined */
	public static String apply(String source) {
		int globals = source.indexOf("uniform Globals");
		if (globals < 0) {
			return source;
		}
		int member = source.indexOf(GLOBALS_LAST_MEMBER, globals);
		int blockEnd = source.indexOf("};", globals);
		if (member < 0 || blockEnd < 0 || member > blockEnd) {
			return source; // Globals block isn't the vanilla layout; leave everything alone
		}
		int insertAt = member + GLOBALS_LAST_MEMBER.length();
		String out = source.substring(0, insertAt) + "\n    float " + UNIFORM + ";" + source.substring(insertAt);
		for (String[] edit : DERIVATIVES) {
			out = out.replace(edit[0], edit[1]);
		}
		return out;
	}
}
