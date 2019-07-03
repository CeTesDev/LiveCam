#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

uniform sampler2D FilterMask;
uniform sampler2D AlphaMask;

uniform int WinWidth;
uniform int WinHeight;

in vec2 TexCoord;

void main() 
{
	vec2 winCoords = vec2(gl_FragCoord.x / WinWidth, gl_FragCoord.y / WinHeight);
	
	FragColor = texture2D(FilterMask, winCoords);

	FragColor.a *= texture2D(AlphaMask, winCoords).a;
	
	if (FragColor.a < 0.01)
	{
		discard;
	}
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}