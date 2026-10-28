#version 330

// Copies raw depth into a float color texture so it can be read back to the CPU
// (OpenGL readback only reads color attachments).

uniform sampler2D DepthSampler;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texelFetch(DepthSampler, ivec2(gl_FragCoord.xy), 0).r, 0.0, 0.0, 1.0);
}
