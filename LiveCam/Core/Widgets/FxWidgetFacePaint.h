#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Common/PointCalcUtil.h>
#include <Graphics/GraphicsModel.h>
#include <FX.h>
#include <Widgets/FaceMapping.h>

using namespace Eigen;

struct HalfFaceParams
{
	friend class HalfFacePaint;
	friend class FxWidgetFacePaint;

	Eigen::Vector4f leftAmbient;
	Eigen::Vector4f rightAmbient;
	Eigen::Vector4f lineAmbient;

	Eigen::Vector4f leftDiffuse;
	Eigen::Vector4f rightDiffuse;
	Eigen::Vector4f lineDiffuse;

	std::string iconImagePath;
	std::string leftTexture;
	std::string rightTexture;
	std::string lineTexture;
	std::string alphaMask;

	float alphaMaskPower;
	float lineWidth = 0;
	float verticalOffset = 0;

	bool texturedEyes = false;
	bool texturedMouth = false;

	HalfFaceParams();

	HalfFaceParams(Eigen::Vector4f leftAmbient, Eigen::Vector4f rightAmbient, Eigen::Vector4f lineAmbient,
		Eigen::Vector4f leftDiffuse, Eigen::Vector4f rightDiffuse, Eigen::Vector4f lineDiffuse,
		std::string leftTexture, std::string rightTexture, std::string lineTexture, float lineWidth,
		std::string alphaMask, float alphaMaskPower = 1);

	void applySuit(boost::property_tree::ptree& suit, bool loadTexturesImmediately);
	void load();

protected:
	GLuint leftTexId = 0;
	GLuint rightTexId = 0;
	GLuint lineTexId = 0;
	GLuint alphaMaskId = 0;
};

class HalfFacePaint
{
	friend class FxWidgetFacePaint;

protected:
	GLuint leftVBO, rightVBO, lineVBO;

	GLuint leftTexCoordVBO = 0, rightTexCoordVBO = 0, lineTexCoordVBO = 0;

	std::vector<std::array<cv::Point2f, FaceMapping::TOTAL_VERTEX_POINTS>> facePoints;
	std::array<std::array<ValueSmoother, TARGET_DETAIL_LIMIT * 2>, MAX_TO_TRACK> facePointsSmoothers;

	std::array<Eigen::Vector2f, 13> faceLeftContour;
	std::array<Eigen::Vector2f, 13> faceRightContour;

	std::array<Eigen::Vector2f, 7> eyeLeftContour;
	std::array<Eigen::Vector2f, 7> eyeRightContour;

	std::array<Eigen::Vector2f, 7> mouthLeftContour;
	std::array<Eigen::Vector2f, 7> mouthRightContour;

	std::array<Eigen::Vector2f, 8> middleTopLines;
	std::array<Eigen::Vector2f, 2> middleBottomLines;

public:
	GLuint textureid = 0;

public:
	void load(bool headTextureMap);
	void transform(TrackingTarget& face, HalfFaceParams &params);
	void draw(TrackingTarget& face, HalfFaceParams &params, std::shared_ptr<cwc::glShader> shader);
	void unload();
};

class FxWidgetFacePaint : public GraphicsModel
{
public:

	HalfFacePaint halfFacePaint;
	std::array<HalfFaceParams, MAX_TO_TRACK> halfFaceParams;

	bool headTextureMap;
	std::array<float, MAX_TO_TRACK> zOffset;
	std::array<bool, MAX_TO_TRACK> match3D;
	std::array<float, 3> bmin = { 0, 0, 0 };
	std::array<float, 3> bmax = { 0, 0, 0 };

	FxWidgetFacePaint();

	void onInputFrameResized() override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;

protected:
	GLuint textureid = 0;
};
