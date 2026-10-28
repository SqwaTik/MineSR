package dev.minesr;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ShaderPatchTest {
	private static final String GLOBALS = """
			layout(std140) uniform Globals {
			    ivec3 CameraBlockPos;
			    vec3 CameraOffset;
			    vec2 ScreenSize;
			    float GlintAlpha;
			    float GameTime;
			    int MenuBlurRadius;
			    int UseRgss;
			};
			""";
	private static final String TERRAIN_BODY = """
			vec4 sampleNearest(sampler2D source, vec2 uv, vec2 pixelSize) {
			    vec2 du = dFdx(uv);
			    vec2 dv = dFdy(uv);
			    return textureGrad(source, uv, du, dv);
			}
			""";

	@Test
	void addsUniformAfterLastGlobalsMemberAndScalesDerivatives() {
		String out = ShaderPatch.apply(GLOBALS + TERRAIN_BODY);
		assertTrue(out.contains("int UseRgss;\n    float MineSRLodScale;\n};"), out);
		assertTrue(out.contains("vec2 du = dFdx(uv) * MineSRLodScale;"));
		assertTrue(out.contains("vec2 dv = dFdy(uv) * MineSRLodScale;"));
	}

	@Test
	void leavesShadersWithoutGlobalsUntouched() {
		assertEquals(TERRAIN_BODY, ShaderPatch.apply(TERRAIN_BODY));
	}

	@Test
	void leavesCustomGlobalsLayoutsUntouched() {
		String custom = GLOBALS.replace("int UseRgss;", "int SomethingElse;") + TERRAIN_BODY;
		assertEquals(custom, ShaderPatch.apply(custom));
	}
}
