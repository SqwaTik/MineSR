#version 330

// Neural upscaler, step 1 of 2. Runs once per LOW-RES pixel: a small trained network looks at
// this sample's neighbourhood and at what the history says was here last frame, and decides
// how sharp to rebuild, how much history to keep, how hard to clamp it, and a memory value.
// temporal_nn.fsh applies the decisions at full resolution.
// Mirrors lowres_decisions() in training/upscaler_torch.py.

uniform sampler2D CurColorSampler;
uniform sampler2D MotionSampler;
uniform sampler2D HistorySampler;

#ifndef HIDDEN
#define HIDDEN 8
#endif
#define HQ (HIDDEN / 4)

layout(std140) uniform UpscaleInfo {
    mat4 InvViewProj;   // current frame, without jitter: clip space -> camera-relative world
    mat4 PrevViewProj;  // previous frame, without jitter: camera-relative world -> clip space
    vec4 CameraDelta;   // xyz: camera position now minus last frame; w: 1 if depth range is [0, 1]
    vec4 Jitter;        // xy: this frame's offset in low-res pixels
    vec4 Sizes;         // xy: low-res size, zw: output size
    vec4 Params;        // x: 1 if history is usable, y: max frames of history to accumulate
};

// Hidden units per layer, set by the mod from the loaded network (multiple of 4).

// Weights, packed by NeuralWeights.java. Row r of a layer with k inputs (k padded to a
// multiple of 4) is vec4s [r * k/4, (r + 1) * k/4).
layout(std140) uniform NetWeights {
    vec4 W1[HIDDEN * 5];  // HIDDEN rows x 20 inputs (17 used)
    vec4 B1[HQ];
    vec4 W2[HIDDEN * HQ]; // HIDDEN rows x HIDDEN
    vec4 B2[HQ];
    vec4 W3[4 * HQ];      // 4 rows x HIDDEN
    vec4 B3;
};

in vec2 texCoord;
out vec4 fragColor; // sharpness, alpha, beta, memory

float sigmoid(float v) {
    return 1.0 / (1.0 + exp(-v));
}

void main() {
    vec2 lowSize = Sizes.xy;
    ivec2 lowMax = ivec2(lowSize) - 1;
    ivec2 p = clamp(ivec2(texCoord * lowSize), ivec2(0), lowMax);

    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    vec3 boxMin = vec3(1e9), boxMax = vec3(-1e9);
    for (int i = 0; i < 9; i++) {
        vec3 c = texelFetch(CurColorSampler, clamp(p + ivec2(i % 3 - 1, i / 3 - 1), ivec2(0), lowMax), 0).rgb;
        m1 += c;
        m2 += c * c;
        boxMin = min(boxMin, c);
        boxMax = max(boxMax, c);
    }
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));

    // Where this sample actually is (its texel center minus the jitter), and where it was.
    vec2 uv = (vec2(p) + 0.5 - Jitter.xy) / lowSize;
    vec4 motion = texelFetch(MotionSampler, p, 0);
    vec2 prevUV = uv + motion.xy;
    bool onScreen = motion.z > 0.5 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    bool haveHistory = Params.x > 0.5 && onScreen;
    vec3 history = vec3(0.0);
    float memory = 0.0;
    if (haveHistory) {
        vec4 hm = texture(HistorySampler, prevUV);
        history = hm.rgb;
        memory = hm.a;
    }
    float have = haveHistory ? 1.0 : 0.0;
    vec3 clipMin = max(boxMin, mean - 1.25 * sigma);
    vec3 clipMax = min(boxMax, mean + 1.25 * sigma);
    vec3 clipped = clamp(history, clipMin, clipMax);

    vec3 h = history - mean, cl = clipped - mean;
    vec3 ad = abs(clipped - history);
    vec4 f[5];
    f[0] = vec4(mean, sigma.r);
    f[1] = vec4(sigma.gb, h.rg);
    f[2] = vec4(h.b, cl);
    f[3] = vec4(memory * have, have, dot(Jitter.xy, Jitter.xy), log(1.0 + length(motion.xy * Sizes.zw)));
    f[4] = vec4(ad.r + ad.g + ad.b, 0.0, 0.0, 0.0);

    // Layer 1: HIDDEN outputs, built 4 at a time. Row r = W1[r * 5 .. r * 5 + 4].
    vec4 h1[HQ];
    for (int q = 0; q < HQ; q++) {
        vec4 s = B1[q];
        for (int k = 0; k < 5; k++) {
            s += vec4(dot(W1[(q * 4) * 5 + k], f[k]), dot(W1[(q * 4 + 1) * 5 + k], f[k]),
                      dot(W1[(q * 4 + 2) * 5 + k], f[k]), dot(W1[(q * 4 + 3) * 5 + k], f[k]));
        }
        h1[q] = max(s, 0.0);
    }
    vec4 h2[HQ];
    for (int q = 0; q < HQ; q++) {
        vec4 s = B2[q];
        for (int k = 0; k < HQ; k++) {
            s += vec4(dot(W2[(q * 4) * HQ + k], h1[k]), dot(W2[(q * 4 + 1) * HQ + k], h1[k]),
                      dot(W2[(q * 4 + 2) * HQ + k], h1[k]), dot(W2[(q * 4 + 3) * HQ + k], h1[k]));
        }
        h2[q] = max(s, 0.0);
    }
    vec4 o = B3;
    for (int k = 0; k < HQ; k++) {
        o += vec4(dot(W3[k], h2[k]), dot(W3[HQ + k], h2[k]), dot(W3[2 * HQ + k], h2[k]), dot(W3[3 * HQ + k], h2[k]));
    }
    fragColor = vec4(1.0 + 11.0 * sigmoid(o.x), sigmoid(o.y), sigmoid(o.z), sigmoid(o.w));
}
