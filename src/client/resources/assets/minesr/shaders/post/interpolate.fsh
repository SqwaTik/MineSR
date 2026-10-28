#version 330

// Frame generation: the frame halfway between the previous world image (N-1) and the current
// one (N), shown before N. Motion comes from the upscaler's motion pass: for each pixel of N,
// where it was in N-1. A surface at x in N was at x + m in N-1, so halfway it is at x + m/2.
// For an output pixel p we look for the x with x + m(x)/2 = p (two fixed-point steps), then take
// the color from both frames. Where they disagree (something appeared or disappeared), the
// current frame wins. The neural version replaces that rule with a trained decision.

uniform sampler2D PrevSampler;
uniform sampler2D CurSampler;
uniform sampler2D MotionSampler;
uniform sampler2D OverlaySampler;

in vec2 texCoord;
out vec4 fragColor;

vec4 motionAt(vec2 uv) {
    return texture(MotionSampler, clamp(uv, vec2(0.0), vec2(1.0)));
}

void main() {
    vec2 p = texCoord;
    vec2 x = p;
    vec4 m = motionAt(x);
    x = p - 0.5 * m.xy;
    m = motionAt(x);
    x = p - 0.5 * m.xy;
    m = motionAt(x);

    vec3 cur = texture(CurSampler, x).rgb;
    vec2 prevUV = x + m.xy;
    bool prevOnScreen = m.z > 0.5 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    vec3 prev = texture(PrevSampler, prevUV).rgb;

    vec3 world = cur;
    if (prevOnScreen) {
        vec3 d = abs(prev - cur);
        float disagreement = max(d.r, max(d.g, d.b));
        world = mix(mix(prev, cur, 0.5), cur, smoothstep(0.08, 0.2, disagreement));
    }
    vec4 overlay = texelFetch(OverlaySampler, ivec2(gl_FragCoord.xy), 0);
    fragColor = vec4(overlay.rgb + world * (1.0 - overlay.a), 1.0);
}
