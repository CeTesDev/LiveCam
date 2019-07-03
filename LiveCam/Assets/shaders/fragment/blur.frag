#version 450 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D Texture;

uniform bool horizontal;
uniform float weight[5] = float[] (0.22, 0.19, 0.12, 0.05, 0.016);

void main()
{             
     vec2 tex_offset = 1.0 / textureSize(Texture, 0); // gets size of single texel
     vec3 result = texture(Texture, TexCoord).rgb * weight[0];
     if(horizontal)
     {
         for(int i = 1; i < 5; ++i)
         {
            result += texture(Texture, TexCoord + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            result += texture(Texture, TexCoord - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
         }
     }
     else
     {
         for(int i = 1; i < 5; ++i)
         {
             result += texture(Texture, TexCoord + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
             result += texture(Texture, TexCoord - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
         }
     }
     FragColor = vec4(result, 1.0);
}