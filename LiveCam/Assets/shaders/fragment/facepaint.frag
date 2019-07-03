#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec2 TexCoord;
in vec3 Position;

struct Line
{
	vec2 A;
	vec2 B;
	float length;
};

uniform vec4 leftColor;
uniform vec4 rightColor;

uniform bool hasTexture;
uniform sampler2D Texture;

uniform bool hasAlphaMask;
uniform sampler2D AlphaMaskTexture;
uniform float alphaMaskPower;

uniform sampler2D OriginFrame;
uniform float faceBrightness;
uniform float frameBrightness;
uniform int frameWidth;
uniform int frameHeight;
uniform vec3 frameLight;

uniform float edgeFadingDistance;
uniform Line edgesLeft[24];
uniform Line edgesRight[24];
uniform Line edgesMiddleTop[7];
uniform Line edgesMiddleBottom[1];

uniform bool isLeft;

varying vec3 vertPos;
varying vec3 vertNV;
varying vec3 vertCol;

float getVertexDistance(vec2 point, vec2 vertex, float min)
{
	vec2 radius = point - vertex;
	float dist = dot(radius, radius);
	if (dist < edgeFadingDistance * edgeFadingDistance && dist < min * min)
	{
		min = sqrt(dist);
	}
	
	return min;
}

float getEdgeDistance(vec2 point, Line edge, float min)
{
	vec2 direction = edge.B - edge.A;
	
	if (dot(direction, point - edge.A) > 0 &&
		dot(-direction, point - edge.B) > 0)
	{
		float dist = abs((direction.y * point.x - direction.x * point.y + edge.B.x * edge.A.y - edge.B.y * edge.A.x) / edge.length);
		if (dist < min)
		{
			min = dist;
		}
	}
	else
	{
		min = getVertexDistance(point, edge.A, min);
	}
	
	return min;
}

float getAlphaScale(vec2 point)
{
	float min = 1e6;
	
	for (int i = 0; i < 24; ++i)
	{
		Line edge = isLeft ? edgesLeft[i] : edgesRight[i];
		
		min = getEdgeDistance(point, edge, min);
	}
	
	min = getVertexDistance(point, isLeft ? edgesLeft[23].B : edgesRight[23].B, min);
	
	return min < edgeFadingDistance ? min / edgeFadingDistance : 1.0;
}

vec4 getMixedColorOnMiddleLine(vec2 point)
{
	float newMin;
	float min = 1e6;
	int index = 0;
	
	for (int i = 0; i < 7; ++i)
	{
		newMin = getEdgeDistance(point, edgesMiddleTop[i], min);
		index = newMin == min ? index : i;
		min = newMin;
	}
	
	newMin = getVertexDistance(point, edgesMiddleTop[6].B, min);
	index = newMin == min ? index : 6;
	min = newMin;
	
	newMin = getEdgeDistance(point, edgesMiddleBottom[0], min);
	index = newMin == min ? index : 7;
	min = newMin;
	
	newMin = getVertexDistance(point, edgesMiddleBottom[0].B, min);
	index = newMin == min ? index : 7;
	min = newMin;
	
	if (min > edgeFadingDistance / 2)
	{
		return isLeft ? leftColor : rightColor;
	}
	
	min = min / edgeFadingDistance + 0.5;
	
	Line edge = index == 7 ? edgesMiddleBottom[0] : edgesMiddleTop[index];
	
	vec4 thisColor, thatColor;
	
	if (cross(vec3(point - edge.A, 0), vec3(edge.B - edge.A, 0)).z > 0)
	{
		thisColor = rightColor;
		thatColor = leftColor;
	}
	else
	{
		thisColor = leftColor;
		thatColor = rightColor;
	}
	
	return vec4(thisColor.rgb * thisColor.a * min + thatColor.rgb * thatColor.a * (1 - min),
		thisColor.a * min + thatColor.a * (1 - min));
}

const float STD_FRAME_BRIGHTNESS = 0.9;

void main()
{
	if (hasTexture)
	{
		FragColor = texture2D(Texture, TexCoord).rgba;
	}
	else
	{
		FragColor = vec4(0, 0, 0, 0);
	}
	
	float alphaFromMask = 1;
	
	if (hasAlphaMask)
	{
		alphaFromMask = 1 - (1 - texture2D(AlphaMaskTexture, TexCoord).r) * alphaMaskPower;
	}
	
	vec4 baseColor = getMixedColorOnMiddleLine(Position.xy);
	vec3 faceColor = texture2D(OriginFrame, vec2(Position.x / frameWidth, Position.y / frameHeight)).rgb;
	
	float frameBrightnessFactor = frameBrightness < STD_FRAME_BRIGHTNESS ? frameBrightness / STD_FRAME_BRIGHTNESS : 1;
	float pixelBrightness = (faceColor.r + faceColor.g + faceColor.b) / 3;
	float brightnessFactor = pixelBrightness / faceBrightness * frameBrightnessFactor;
	
	FragColor.rgb = (baseColor.rgb * baseColor.a * (1 - FragColor.a) + FragColor.rgb * FragColor.a) * brightnessFactor; // * frameLight * frameLight * frameLight * frameLight;
	
	FragColor.a = (baseColor.a + FragColor.a) * alphaFromMask * getAlphaScale(Position.xy);
	float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
    else
        BrightColor = vec4(0, (faceColor.r + faceColor.g + faceColor.b) / 3, 0, 1);
}