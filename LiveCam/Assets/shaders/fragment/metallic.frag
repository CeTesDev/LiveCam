#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;
in vec3 ReflectVector;

uniform sampler2D Texture;
uniform sampler2D SpecularTexture;
uniform samplerCube CubeMap;

uniform vec4 lightPos;
uniform vec4 cameraPos;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
uniform float specularPower;

void specularModel( vec3 norm, out vec3 ambAndDiff, out vec3 spec ) 
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
    
    specularModel(norm, ambAndDiff, spec);

    BrightColor = texture2D(Texture, TexCoord).rgba;
    
    BrightColor.r = BrightColor.r * ambAndDiff.r + spec.r;
    BrightColor.g = BrightColor.g * ambAndDiff.g + spec.g;
    BrightColor.b = BrightColor.b * ambAndDiff.b + spec.b;
    
    vec4 reflectColor = texture(CubeMap, ReflectVector);
    FragColor = mix(BrightColor, reflectColor, 0.6);
}
