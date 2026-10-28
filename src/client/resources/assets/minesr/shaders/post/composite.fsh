#version 330

// Frame generation: the HUD, hand and GUI are drawn onto their own transparent layer (colors
// premultiplied by alpha, as vanilla's blending produces on a cleared target). This puts that
// layer back on top of the world image.

uniform sampler2D WorldSampler;
uniform sampler2D OverlaySampler;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 world = texelFetch(WorldSampler, p, 0).rgb;
    vec4 overlay = texelFetch(OverlaySampler, p, 0);
    fragColor = vec4(overlay.rgb + world * (1.0 - overlay.a), 1.0);
}
