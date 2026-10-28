package dev.minesr.client.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.SkyRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * The sky renderer keeps the target it was created with. Whether that was the low-res or the
 * full-res one depends on timing, and the low-res one can be freed later. Look it up every
 * time instead, so the sky always lands where the rest of the world is being drawn.
 */
@Mixin(SkyRenderer.class)
public abstract class SkyRendererMixin {
	@Redirect(method = "*", at = @At(value = "FIELD",
			target = "Lnet/minecraft/client/renderer/SkyRenderer;renderTarget:Lcom/mojang/blaze3d/pipeline/RenderTarget;",
			opcode = org.objectweb.asm.Opcodes.GETFIELD))
	private RenderTarget minesr$currentTarget(SkyRenderer self) {
		return Minecraft.getInstance().gameRenderer.mainRenderTarget();
	}
}
