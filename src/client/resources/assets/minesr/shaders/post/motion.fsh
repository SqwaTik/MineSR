#version 330

// Runs once per LOW-RES pixel: where was this surface last frame?
// Doing this here instead of per output pixel saves 9 depth reads per output pixel.
// Output: xy = screen-space motion (previous uv - current uv), z = 1 if the previous
// position was on screen.

uniform sampler2D CurDepthSampler;

layout(std140) uniform UpscaleInfo {
    mat4 InvViewProj;
    mat4 PrevViewProj;
    vec4 CameraDelta;   // xyz: camera movement since last frame; w: 1 if depth range is [0, 1]
    vec4 Jitter;
    vec4 Sizes;         // xy: low-res size, zw: output size
    vec4 Params;
};

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 lowSize = Sizes.xy;
    ivec2 lowMax = ivec2(lowSize) - 1;
    ivec2 p = clamp(ivec2(texCoord * lowSize), ivec2(0), lowMax);

    // Closest depth nearby (reversed depth: larger is closer), so thin edges move with the
    // object in front instead of smearing into the background.
    float closestDepth = 0.0;
    for (int i = 0; i < 9; i++) {
        ivec2 t = clamp(p + ivec2(i % 3 - 1, i / 3 - 1), ivec2(0), lowMax);
        closestDepth = max(closestDepth, texelFetch(CurDepthSampler, t, 0).r);
    }

    // This texel shows the scene at its center minus the jitter.
    vec2 uv = (vec2(p) + 0.5 - Jitter.xy) / lowSize;
    float ndcZ = CameraDelta.w > 0.5 ? closestDepth : closestDepth * 2.0 - 1.0;
    vec4 world = InvViewProj * vec4(uv * 2.0 - 1.0, ndcZ, 1.0);
    // Homogeneous add: for the sky (w = 0) only rotation matters, not camera movement.
    vec4 prevClip = PrevViewProj * (world + vec4(CameraDelta.xyz * world.w, 0.0));
    vec2 prevUV = prevClip.xy / prevClip.w * 0.5 + 0.5;
    bool onScreen = prevClip.w > 0.0 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    fragColor = vec4(prevUV - uv, onScreen ? 1.0 : 0.0, 0.0);
}
