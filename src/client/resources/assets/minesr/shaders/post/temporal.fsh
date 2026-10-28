#version 330

// Temporal upscale: rebuild a full-resolution image from the current jittered low-res frame
// plus the previous full-resolution result, moved to where each pixel is now.
//
// This is the hand-tuned baseline. The neural version keeps the same inputs and replaces
// the "how much history to trust" rules (clip box, blend weight) with a trained network.

uniform sampler2D CurColorSampler;
uniform sampler2D MotionSampler; // from motion.fsh, one texel per low-res pixel
uniform sampler2D HistorySampler;

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
    vec2 x = texCoord * lowSize; // this output pixel, in low-res pixel units

    // Low-res texel i shows the scene at i + 0.5 - Jitter. Find the one nearest to x.
    vec2 nearest = floor(x + Jitter.xy);

    vec3 colors[9];
    float dist2[9];
    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    vec3 boxMin = vec3(1e9), boxMax = vec3(-1e9);

    for (int i = 0; i < 9; i++) {
        vec2 texel = nearest + vec2(i % 3 - 1, i / 3 - 1);
        ivec2 t = clamp(ivec2(texel), ivec2(0), lowMax);
        vec3 c = texelFetch(CurColorSampler, t, 0).rgb;
        vec2 d = (texel + 0.5 - Jitter.xy) - x;
        colors[i] = c;
        dist2[i] = dot(d, d);
        m1 += c;
        m2 += c * c;
        boxMin = min(boxMin, c);
        boxMax = max(boxMax, c);
    }

    // Where was this surface last frame? Computed once per low-res pixel by motion.fsh.
    vec4 motion = texelFetch(MotionSampler, clamp(ivec2(nearest), ivec2(0), lowMax), 0);
    vec2 prevUV = texCoord + motion.xy;
    bool onScreen = motion.z > 0.5 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    float history_n = 0.0;
    vec3 history = vec3(0.0);
    bool haveHistory = Params.x > 0.5 && onScreen;
    if (haveHistory) {
        history = sampleHistory(prevUV, Sizes.zw, history_n);
    }

    // Rebuild this frame's color here. With little history, a wide filter fills the pixel from
    // its neighbours; with plenty, a narrow one adds a near point-sample at the jittered
    // position, which is what lets accumulation converge to a sharp image instead of a blur.
    float sharpness = mix(2.3, 9.0, clamp(history_n / 4.0, 0.0, 1.0));
    vec3 colorSum = vec3(0.0);
    float weightSum = 0.0;
    float nearestWeight = 0.0;
    for (int i = 0; i < 9; i++) {
        float w = exp(-sharpness * dist2[i]);
        colorSum += colors[i] * w;
        weightSum += w;
        nearestWeight = max(nearestWeight, w);
    }
    vec3 current = colorSum / max(weightSum, 1e-5);

    if (!haveHistory) {
        fragColor = vec4(current, nearestWeight);
        return;
    }

    // Rectify: history may only contain colors plausible for this neighborhood now.
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    vec3 clipMin = max(boxMin, mean - 1.25 * sigma);
    vec3 clipMax = min(boxMax, mean + 1.25 * sigma);
    vec3 clipped = clamp(history, clipMin, clipMax);

    // The more clipping changed it, the less that history is worth.
    float change = length(clipped - history) / (length(mean) + 0.05);
    history_n *= clamp(1.0 - change * 4.0, 0.1, 1.0);

    // Running average: this frame counts in proportion to how well a sample covered the pixel.
    float n = min(history_n + nearestWeight, Params.y);
    float alpha = nearestWeight / max(n, 1e-4);
    fragColor = vec4(mix(clipped, current, alpha), n);
}
