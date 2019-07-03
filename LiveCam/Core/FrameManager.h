#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <map>

struct FrameRenderBuffer
{
	GLuint frameBuffer = 0;
	GLuint depthRenderBuffer = 0;
	GLuint textureId = 0;
	GLuint texture2Id = 0;
	GLuint filterMaskId = 0;
	GLuint alphaMaskId = 0;
	size_t width = 0;
	size_t height = 0;
};

enum class ColorBuffer
{
	COLOR_1 = GL_COLOR_ATTACHMENT0,
	COLOR_2 = GL_COLOR_ATTACHMENT1,
	FILTER_MASK = GL_COLOR_ATTACHMENT2,
	ALPHA_MASK = GL_COLOR_ATTACHMENT3,
	GLOW_MASK = GL_COLOR_ATTACHMENT4,
	SKYBOX = GL_COLOR_ATTACHMENT5
};

class FrameManager
{

public:
	FrameManager();
	~FrameManager();

	size_t currentFBOindex;
	size_t currentBuffer;

	size_t AddFrameBuffer(size_t width, size_t height);

	void RemoveFrameBuffer(size_t frameIndex);

	void SwitchToFrameBuffer(size_t frameIndex);

	void SwitchToDrawBuffer(size_t bufferIndex);

	void SwitchToScreen();

	FrameRenderBuffer getFrameRenderBuffer();

	size_t getCurrentFBOindex();
	size_t getCurrentDrawBuffer();

protected:
	static size_t nextFrameIndex;

	std::map<size_t, FrameRenderBuffer> frameBufferMap;

	void DeleteFrameBuffer(FrameRenderBuffer& frameBuffer);
};
