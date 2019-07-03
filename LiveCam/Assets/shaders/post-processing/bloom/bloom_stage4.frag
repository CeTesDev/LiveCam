#version 450 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D Texture;
uniform sampler2D BloomTexture;

void main()
{             
    const float gamma = 2.2;
    vec3 hdrColor = texture(Texture, TexCoord).rgb;      
    vec3 bloomColor = texture(BloomTexture, TexCoord).rgb;
    hdrColor += bloomColor; // additive blending
    // tone mapping
    vec3 result = vec3(1.0) - exp(-hdrColor * 2.0);
    // also gamma correct while we're at it       
    result = pow(result, vec3(1.0 / gamma));
    FragColor = vec4(result, 1.0);
}