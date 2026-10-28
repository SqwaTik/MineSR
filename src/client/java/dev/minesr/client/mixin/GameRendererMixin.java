package dev.minesr.client.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import dev.minesr.client.MineSRClient;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.GameRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Wraps only the world part of {@code renderLevel}: projection jitter, the low-res target
 * swap, and the upscale right after. Hand, screen effects and GUI stay at full resolution.
 */
@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {
	@Shadow
	@Final
	private RenderTarget mainRenderTarget;

	@Shadow
	@Final
	private GameRenderState gameRenderState;

	@ModifyArg(
			method = "renderLevel",
			at = @At(value = "INVOKE",
					target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/blaze3d/buffers/GpuBufferSlice;",
					ordinal = 0))
	private Matrix4f minesr$jitter(Matrix4f projection) {
		CameraRenderState camera = gameRenderState.levelRenderState.cameraRenderState;
		return MineSRClient.upscaler().jitterProjection(
				projection, camera.viewRotationMatrix, camera.pos, mainRenderTarget.width, mainRenderTarget.height);
	}

	@Inject(method = "renderLevel", at = @At(value = "INVOKE", target = WORLD_RENDER))
	private void minesr$beginWorld(DeltaTracker deltaTracker, CallbackInfo ci) {
		MineSRClient.presentPendingFrameIfDue();
		MineSRClient.upscaler().beginLevel();
	}

	@Inject(method = "renderLevel", at = @At(value = "INVOKE", target = WORLD_RENDER, shift = At.Shift.AFTER))
	private void minesr$endWorld(DeltaTracker deltaTracker, CallbackInfo ci) {
		MineSRClient.upscaler().endLevel(mainRenderTarget);
		MineSRClient.frameGen().beginOverlay(mainRenderTarget); // hand, effects and GUI go onto their own layer
		MineSRClient.presentPendingFrameIfDue();
	}

	@Inject(method = "render", at = @At("HEAD"))
	private void minesr$frameStart(DeltaTracker deltaTracker, boolean advanceGameTime, CallbackInfo ci) {
		MineSRClient.presentPendingFrameIfDue();
	}

	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/gui/render/GuiRenderer;render()V"))
	private void minesr$beforeGui(DeltaTracker deltaTracker, boolean advanceGameTime, CallbackInfo ci) {
		MineSRClient.frameGen().beforeGui();
		MineSRClient.presentPendingFrameIfDue(true);
	}

	/** Everything that draws the world asks for the main target through this method. */
	@Inject(method = "mainRenderTarget", at = @At("HEAD"), cancellable = true)
	private void minesr$lowResDuringWorld(CallbackInfoReturnable<RenderTarget> cir) {
		if (MineSRClient.upscaler().isLevelActive()) {
			cir.setReturnValue(MineSRClient.upscaler().lowTarget());
		} else if (MineSRClient.frameGen().isOverlayActive()) {
			cir.setReturnValue(MineSRClient.frameGen().overlayTarget());
		}
	}

	private static final String WORLD_RENDER = "Lnet/minecraft/client/renderer/LevelRenderer;render("
			+ "Lcom/mojang/blaze3d/resource/GraphicsResourceAllocator;Lnet/minecraft/client/DeltaTracker;Z"
			+ "Lnet/minecraft/client/renderer/state/level/CameraRenderState;Lorg/joml/Matrix4fc;"
			+ "Lcom/mojang/blaze3d/buffers/GpuBufferSlice;Lorg/joml/Vector4f;Z)V";
}
