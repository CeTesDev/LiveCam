#version 450
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube CubeMap;
uniform sampler2D Texture;

void main()
{    
    FragColor = texture(CubeMap, TexCoords);
}