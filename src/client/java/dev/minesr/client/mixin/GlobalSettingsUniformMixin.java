package dev.minesr.client.mixin;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.systems.RenderSystem;
import dev.minesr.client.MineSRClient;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.renderer.GlobalSettingsUniform;
import net.minecraft.world.phys.Vec3;
import org.lwjgl.system.MemoryStack;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.nio.ByteBuffer;

/**
 * Feeds {@code MineSRLodScale}, the float {@link dev.minesr.ShaderPatch} appends to the
 * Globals block. Vanilla's block ends at byte 56; the extra float sits right after it.
 */
@Mixin(GlobalSettingsUniform.class)
public abstract class GlobalSettingsUniformMixin {
	private static final int LOD_SCALE_OFFSET = GlobalSettingsUniform.UBO_SIZE;
	/** Vanilla size plus one float, rounded up to the 16-byte block alignment. */
	private static final int PATCHED_SIZE = (GlobalSettingsUniform.UBO_SIZE + 4 + 15) & ~15;

	@Shadow
	@Final
	private GpuBuffer buffer;

	@ModifyArg(method = "<init>",
			at = @At(value = "INVOKE",
					target = "Lcom/mojang/blaze3d/systems/GpuDevice;createBuffer(Ljava/util/function/Supplier;IJ)Lcom/mojang/blaze3d/buffers/GpuBuffer;"),
			index = 2)
	private long minesr$roomForLodScale(long size) {
		return Math.max(size, PATCHED_SIZE);
	}

	@Inject(method = "update", at = @At("TAIL"))
	private void minesr$writeLodScale(int width, int height, double glintAlpha, long gameTime, DeltaTracker deltaTracker,
			int menuBlurRadius, Vec3 cameraPos, boolean useRgss, CallbackInfo ci) {
		float scale = MineSRClient.upscaler() == null ? 1.0f : MineSRClient.upscaler().lodScale();
		try (MemoryStack stack = MemoryStack.stackPush()) {
			ByteBuffer data = stack.malloc(4);
			data.putFloat(0, scale);
			RenderSystem.getDevice().createCommandEncoder().writeToBuffer(buffer.slice(LOD_SCALE_OFFSET, 4), data);
		}
	}
}
