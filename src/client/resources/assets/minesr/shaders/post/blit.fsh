#version 330

// Bilinear copy, used to show the high-resolution capture on screen while recording.

uniform sampler2D InSampler;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(InSampler, texCoord).rgb, 1.0);
}
