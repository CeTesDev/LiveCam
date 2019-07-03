#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <FXModel.h>
#include <Graphics/GraphicsLoader.h>
#include <Tracking/ObjectTracker.h>
#include <FX.h>
#include <Common/Resolutions.h>
#include "FrameManager.h"

extern double cameraDefaultVerticalAngle;
extern double VulkanzNear;
extern double VulkanzFar;

extern FrameManager frameManager;

enum class AntiAliasing
{
	NONE,
	SSAA_4X
};

extern const std::map<std::string, AntiAliasing> AntiAliasingMap;

class VulkanDataObject
{
public:

	int numTriangles;

	size_t material_id;
	std::string material;

	GLuint vb = 0;
	GLuint nb = 0;
	GLuint cb = 0;
	GLuint tb = 0;

	GLuint tangentb = 0;
	GLuint bitangentb = 0;

	std::string name;

	float minimumZ = 0;

};

class VulkanMesh {
public:
	int rows, cols;
	std::vector<cv::Point2f>		vertices;
	std::vector<cv::Point2f>		uvs;
	std::vector<ushort>				idxs;

	VulkanMesh() {}
	~VulkanMesh() {}

	VulkanMesh(const VulkanMesh& t) {
		rows = t.rows;
		cols = t.cols;
		vertices.resize(t.vertices.size());
		uvs.resize(t.uvs.size());
		idxs.resize(t.idxs.size());
		std::copy(t.vertices.begin(), t.vertices.end(), vertices.begin());
		std::copy(t.uvs.begin(), t.uvs.end(), uvs.begin());
		std::copy(t.idxs.begin(), t.idxs.end(), idxs.begin());
	}

	void release() {
		vertices.clear();
		uvs.clear();
		idxs.clear();
	}
};

class VulkanMain
{
public:
	int frame_width;
	int frame_height;
	GLFWwindow* window;
	VulkanMesh mesh;

	Matrix4f textureProjection;
	Matrix4f particlesProjection;
	VulkanDataObject textureQuad;
	std::shared_ptr<cwc::glShader> textureShader;
	std::shared_ptr<cwc::glShader> alphaMaskShader;
	std::shared_ptr<cwc::glShader> antiAliasingShader;

	FX *currentEffect;
	bool isTracking;

	size_t cameraFrameIndex;

	void drawForeground(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces);

	cv::Mat drawBackground(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces);
	
	cv::Mat addTrackerValues(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces);

	void updateAllTextures(FilterModule *module);
	void updateTexture(FilterModule *module, int targetIndex);

	bool loadEffect(FX *effect);
			
	void close();

	int init(AntiAliasing antiAliasing);

	void initAntiAliasing4XShader();
	
	void RecreateSimpleTexture(int width, int height);

	void PostProcessing();

	void DrawTexture(GLuint texId, std::shared_ptr<cwc::glShader> shader);

	typedef std::pair<GLuint, std::string> TextureUniform;
	void DrawTextures(std::vector<TextureUniform> textures, std::shared_ptr<cwc::glShader> shader);

};