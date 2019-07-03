#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D Texture;

uniform vec4 lightPos;
uniform vec4 cameraPos;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
uniform float specularPower;

void main()
{           
    FragColor = texture(Texture, TexCoord).rgba;
    // check whether result is higher than some threshold, if so, output as bloom threshold color
    float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
    else
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}