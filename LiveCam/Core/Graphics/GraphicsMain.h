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

#include <Common/Resolutions.h>
#include <Graphics/GraphicsCommon.h>
#include <Tracking/Detector.h>
#include <Tracking/TrackingTarget.h>
#include <Utils/DebugTools.h>
#include <FX.h>
#include <FrameManager.h>

extern double cameraDefaultVerticalAngle;
extern double GraphicszNear;
extern double GraphicszFar;

extern FrameManager frameManager;

enum class AntiAliasing
{
	NONE,
	SSAA_4X
};

extern const std::map<std::string, AntiAliasing> AntiAliasingMap;

class GraphicsMain
{
public:
	int frame_width;
	int frame_height;
	GLFWwindow* window;
	Mesh3D mesh;

	Matrix4f textureProjection;
	Matrix4f modelViewProjection;
	Matrix4f particlesProjection;
	GraphicsData textureQuad;
	std::shared_ptr<cwc::glShader> textureShader;
	std::shared_ptr<cwc::glShader> alphaMaskShader;
	std::shared_ptr<cwc::glShader> antiAliasingShader;
	std::shared_ptr<cwc::glShader> cubemapShader;
	std::shared_ptr<cwc::glShader> skyboxShader;

	GLuint cubeVAO, cubeVBO;
	GLuint skyboxVAO, skyboxVBO;
	GLuint cubemapTexture;

	FX *currentEffect;
	bool isTracking;

	size_t cameraFrameIndex;

	void initSkybox();
	void drawForeground(cv::Mat frame, std::vector<TrackingTarget>& targets);

	cv::Mat drawBackground(cv::Mat frame, std::vector<TrackingTarget>& targets);
	
	cv::Mat addTrackerValues(cv::Mat frame, std::vector<TrackingTarget>& targets);

	void updateAllTextures(FilterModule *module);
	void updateTexture(FilterModule *module, int targetIndex);

	bool loadEffect(FX *effect);
			
	void close();

	int init(AntiAliasing antiAliasing);

	void initAntiAliasing4XShader();
	
	void RecreateSimpleTexture(int width, int height);

	void PostProcessing();
	void ApplySkybox();

	void DrawTexture(GLuint texId, std::shared_ptr<cwc::glShader> shader);

	typedef std::pair<GLuint, std::string> TextureUniform;
	void DrawTextures(std::vector<TextureUniform> textures, std::shared_ptr<cwc::glShader> shader);
	void SwapColorBuffer(FrameRenderBuffer frb, GLuint sb, GLuint tb);

};