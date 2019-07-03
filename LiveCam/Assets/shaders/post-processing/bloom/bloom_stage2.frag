#version 450 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D Texture;

void main()
{
	vec4 textureColor = texture2D(Texture, TexCoord).rgba;
    
    textureColor.r = textureColor.r * 2.0;
    textureColor.g = textureColor.g * 3.0;
    textureColor.b = textureColor.b * 4.0;
    
	FragColor = textureColor; 
	//FragColor = vec4(vec3(0.0, 0.0, 15.0), 1.0);
    float brightness = dot(textureColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(textureColor.rgb, 1.0);
    else
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}