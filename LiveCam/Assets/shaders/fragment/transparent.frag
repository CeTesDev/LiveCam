#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec2 TexCoord;

uniform sampler2D Texture;

void main()
{
	FragColor = vec4(0, 0, 0, 0);
    BrightColor = vec4(0, 0, 0, 0);
}