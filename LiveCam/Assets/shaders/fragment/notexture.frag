#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform vec4 materialColor;
uniform vec4 lightPos;
uniform vec4 cameraPos;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
uniform float specularPower;
uniform sampler2D Texture;

void applyLighting( vec3 norm, out vec3 ambAndDiff, out vec3 spec ) 
{
    vec3 lightDirection = lightPos.w == 0 ? normalize(vec3(lightPos)) : normalize(vec3(lightPos) - Position);
    vec3 cameraDirection =  cameraPos.w == 0 ? normalize(vec3(cameraPos)) : normalize(vec3(cameraPos) - Position);
    vec3 lightReflection = reflect( -lightDirection, norm );

    float sDotN = max( dot(lightDirection,norm), 0.0 );
    
    vec3 diffuse = diffuseLight * sDotN;
    ambAndDiff = ambientLight + diffuse;
    
    spec = vec3(0.0);
    if(sDotN > 0.0)
    {
        spec = specularLight * pow(max( dot(lightReflection, cameraDirection), 0.0 ), specularPower);
        // spec is NAN, if dot == 0 && specularPower <= 0
    }
}

void main()
{
    vec3 ambAndDiff, spec;
    vec3 norm = normalize(Normal);  
    
    applyLighting(norm, ambAndDiff, spec);

    vec4 textureColor = materialColor;
    
    textureColor.r = textureColor.r * ambAndDiff.r + spec.r;
    textureColor.g = textureColor.g * ambAndDiff.g + spec.g;
    textureColor.b = textureColor.b * ambAndDiff.b + spec.b;
    
    FragColor = textureColor;
    float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
    else
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}