#version 330

// Contrast-adaptive sharpening of the upscaled image into the game's main target.
// Sharpens flat detail, backs off near strong edges so it doesn't ring.

uniform sampler2D InSampler;

layout(std140) uniform SharpenInfo {
    vec4 Sharpen; // x: strength 0..1
};

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 lim = textureSize(InSampler, 0) - 1;
    vec3 c = texelFetch(InSampler, p, 0).rgb;
    vec3 n = texelFetch(InSampler, clamp(p + ivec2(0, 1), ivec2(0), lim), 0).rgb;
    vec3 s = texelFetch(InSampler, clamp(p - ivec2(0, 1), ivec2(0), lim), 0).rgb;
    vec3 e = texelFetch(InSampler, clamp(p + ivec2(1, 0), ivec2(0), lim), 0).rgb;
    vec3 w = texelFetch(InSampler, clamp(p - ivec2(1, 0), ivec2(0), lim), 0).rgb;

    vec3 mn = min(c, min(min(n, s), min(e, w)));
    vec3 mx = max(c, max(max(n, s), max(e, w)));
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, 1e-4), 0.0, 1.0));
    vec3 k = -amp / mix(8.0, 5.0, clamp(Sharpen.x, 0.0, 1.0)) * step(0.001, Sharpen.x);
    vec3 result = (c + (n + s + e + w) * k) / (1.0 + 4.0 * k);
    fragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
