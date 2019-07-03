#version 450 core
out vec4 FragColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D Texture;
uniform sampler2D SpecularTexture;
uniform sampler2D EmissiveTexture;
uniform vec4 lightPos;
uniform vec4 cameraPos;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
uniform float specularPower;

void main()
{
    // ambient
    vec3 color = texture(Texture, TexCoord).rgb;
    vec3 ambient = ambientLight * color;
    
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(vec3(lightPos) - Position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diffuseLight * diff * color;  
    
    // specular
    vec3 viewDir = normalize(vec3(cameraPos) - Position);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), specularPower);
    vec3 specular = specularLight * spec * texture(SpecularTexture, TexCoord).rgb;  
    
      // emission
    vec3 emission = texture(EmissiveTexture, TexCoord).rgb;
        
    vec3 hdrColor = ambient + diffuse + specular + emission;
    FragColor = vec4(hdrColor, 1.0);
} 