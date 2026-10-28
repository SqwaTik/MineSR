#version 330

// Temporal upscale: rebuild a full-resolution image from the current jittered low-res frame
// plus the previous full-resolution result, moved to where each pixel is now.
//
// Neural upscaler, step 2 of 2: the same pass as temporal.fsh, but the per-pixel decisions
// come from decide.fsh (one network run per low-res pixel), looked up at the nearest sample.
// Mirrors step_lowres() in training/upscaler_torch.py.

uniform sampler2D CurColorSampler;
uniform sampler2D MotionSampler; // from motion.fsh, one texel per low-res pixel
uniform sampler2D HistorySampler;
uniform sampler2D DecisionSampler; // from decide.fsh: sharpness, alpha, beta, memory

layout(std140) uniform UpscaleInfo {
    mat4 InvViewProj;   // current frame, without jitter: clip space -> camera-relative world
    mat4 PrevViewProj;  // previous frame, without jitter: camera-relative world -> clip space
    vec4 CameraDelta;   // xyz: camera position now minus last frame; w: 1 if depth range is [0, 1]
    vec4 Jitter;        // xy: this frame's offset in low-res pixels
    vec4 Sizes;         // xy: low-res size, zw: output size
    vec4 Params;        // x: 1 if history is usable, y: max frames of history to accumulate
};

in vec2 texCoord;
out vec4 fragColor;

// Catmull-Rom from 5 bilinear taps: the 4 corner taps carry almost no weight, so they're
// dropped and the rest renormalized. Sharp enough that history doesn't blur over time.
vec3 sampleHistory(vec2 uv, vec2 size, out float weightOut) {
    vec2 pos = uv * size;
    vec2 center = floor(pos - 0.5) + 0.5;
    vec2 f = pos - center;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = (center - 1.0) / size;
    vec2 t3 = (center + 2.0) / size;
    vec2 t12 = (center + w2 / w12) / size;

    vec4 middle = texture(HistorySampler, t12);
    float wc = w12.x * w12.y;
    float wl = w0.x * w12.y, wr = w3.x * w12.y, wb = w12.x * w0.y, wt = w12.x * w3.y;
    vec3 sum = middle.rgb * wc
            + texture(HistorySampler, vec2(t0.x, t12.y)).rgb * wl
            + texture(HistorySampler, vec2(t3.x, t12.y)).rgb * wr
            + texture(HistorySampler, vec2(t12.x, t0.y)).rgb * wb
            + texture(HistorySampler, vec2(t12.x, t3.y)).rgb * wt;
    // Accumulated-frame count lives in alpha; the middle tap is close enough.
    weightOut = middle.a;
    return max(sum / (wc + wl + wr + wb + wt), vec3(0.0));
}

void main() {
    vec2 lowSize = Sizes.xy;
    ivec2 lowMax = ivec2(lowSize) - 1;
    vec2 x = texCoord * lowSize;
    vec2 nearest = floor(x + Jitter.xy);
    ivec2 nearestTexel = clamp(ivec2(nearest), ivec2(0), lowMax);
    vec4 decision = texelFetch(DecisionSampler, nearestTexel, 0);
    float sharpness = decision.x;

    vec3 colors[9];
    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    vec3 boxMin = vec3(1e9), boxMax = vec3(-1e9);
    vec3 colorSum = vec3(0.0);
    float weightSum = 0.0;
    float nearestWeight = 0.0;
    for (int i = 0; i < 9; i++) {
        vec2 texel = nearest + vec2(i % 3 - 1, i / 3 - 1);
        vec3 c = texelFetch(CurColorSampler, clamp(ivec2(texel), ivec2(0), lowMax), 0).rgb;
        vec2 d = (texel + 0.5 - Jitter.xy) - x;
        float w = exp(-sharpness * dot(d, d));
        colorSum += c * w;
        weightSum += w;
        nearestWeight = max(nearestWeight, w);
        m1 += c;
        m2 += c * c;
        boxMin = min(boxMin, c);
        boxMax = max(boxMax, c);
    }
    vec3 current = colorSum / max(weightSum, 1e-5);

    vec4 motion = texelFetch(MotionSampler, nearestTexel, 0);
    vec2 prevUV = texCoord + motion.xy;
    bool onScreen = motion.z > 0.5 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    if (Params.x < 0.5 || !onScreen) {
        fragColor = vec4(current, decision.w);
        return;
    }
    float unusedMemory;
    vec3 history = sampleHistory(prevUV, Sizes.zw, unusedMemory);
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    vec3 clipped = clamp(history, max(boxMin, mean - 1.25 * sigma), min(boxMax, mean + 1.25 * sigma));
    vec3 rectified = history + (clipped - history) * decision.z;
    float alpha = decision.y * nearestWeight; // a sample far from this pixel counts for less
    fragColor = vec4(rectified + (current - rectified) * alpha, decision.w);
}
