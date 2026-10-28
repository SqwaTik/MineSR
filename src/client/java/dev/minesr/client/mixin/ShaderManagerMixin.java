package dev.minesr.client.mixin;

import dev.minesr.ShaderPatch;
import net.minecraft.client.renderer.ShaderManager;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/** Adds the LOD-scale uniform to shader sources as they are loaded. See {@link ShaderPatch}. */
@Mixin(ShaderManager.class)
public abstract class ShaderManagerMixin {
	@ModifyArg(method = "loadShader",
			at = @At(value = "INVOKE",
					target = "Lcom/google/common/collect/ImmutableMap$Builder;put(Ljava/lang/Object;Ljava/lang/Object;)Lcom/google/common/collect/ImmutableMap$Builder;"),
			index = 1)
	private static Object minesr$patchSource(Object source) {
		if (!(source instanceof String text)) {
			return source;
		}
		String patched = ShaderPatch.apply(text);
		if (patched.contains("dFdx(uv) * " + ShaderPatch.UNIFORM)) {
			LoggerFactory.getLogger("minesr").info("MineSR: texture LOD bias added to a world shader");
		}
		return patched;
	}
}
