#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D Texture;

float Hash( vec2 p)
{
    vec3 p2 = vec3(p.xy,1.0);
    return fract(sin(dot(p2,vec3(37.1,61.7, 12.4)))*758.5453123);
}


float noise(in vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f *= f * (3.0-2.0*f);


    return mix(mix(Hash(i + vec2(0.,0.)), Hash(i + vec2(1.,0.)),f.x),
               mix(Hash(i + vec2(0.,1.)), Hash(i + vec2(1.,1.)),f.x),
               f.y);
}


float fbm(vec2 p)
{
     float v = 0.0;
     v += noise(p*1.)*.100;
     v += noise(p*2.)*.25;
     v += noise(p*4.)*.125;
     v += noise(p*8.)*.0625;
     return v;
}


void main( void ) 
{
	vec2 uv = ( gl_FragCoord.xy / Position.xy ) * 2.0 - 1.0;
	uv.x *= Position.x/Position.y;


	vec3 finalColor = vec3( 0.0 );
	for( int i=1; i < 8; ++i )
	{
	float t = abs(1.0 / ((uv.x + fbm( uv + time/float(i)))*75.));
	finalColor +=  t * vec3( float(i) * 0.1 +0.1, 0.8, 2.0 );
	}

	FragColor = vec4( finalColor, 2.0 );
	float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
    else
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}