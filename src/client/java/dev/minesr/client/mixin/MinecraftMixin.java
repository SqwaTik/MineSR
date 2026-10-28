package dev.minesr.client.mixin;

import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuSurface;
import com.mojang.blaze3d.textures.GpuTextureView;
import dev.minesr.client.MineSRClient;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Frame generation: decides which image goes to the screen at the end of each frame. */
@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
	@Shadow
	@Final
	private GpuSurface windowSurface;

	/** Before vanilla fetches the main image to present: stop redirecting it to the overlay. */
	@Inject(method = "renderFrame", at = @At(value = "INVOKE",
			target = "Lcom/mojang/blaze3d/systems/GpuSurface;isAcquired()Z", ordinal = 1))
	private void minesr$endOverlay(boolean advanceGameTime, CallbackInfo ci) {
		MineSRClient.frameGen().endOverlay();
	}

	@Redirect(method = "renderFrame", at = @At(value = "INVOKE",
			target = "Lcom/mojang/blaze3d/systems/GpuSurface;blitFromTexture(Lcom/mojang/blaze3d/systems/CommandEncoder;Lcom/mojang/blaze3d/textures/GpuTextureView;)V"))
	private void minesr$present(GpuSurface surface, CommandEncoder encoder, GpuTextureView mainColor) {
		Minecraft mc = (Minecraft) (Object) this;
		GpuTextureView shown = MineSRClient.frameGen().beforePresent(surface, mainColor, mc.gameRenderer.mainRenderTarget());
		surface.blitFromTexture(encoder, shown);
	}
}
