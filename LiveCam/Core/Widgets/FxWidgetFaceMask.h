#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Common/CommonClasses.h>
#include <Common/PointCalcUtil.h>
#include <Graphics/GraphicsModel.h>
#include <FX.h>
#include <Widgets/FxModSkeleton.h>

using namespace Eigen;

class FxWidgetFaceMask : public GraphicsModel
{
public:
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	std::string modelPath;
	std::string bonesPath;
	std::string weightsPath;

	bool bonesStretching;

	Skeleton skeleton;
	std::vector<ObjectData> initialMeshes;
	std::vector<ObjectData> transformedMeshes;

	std::array<float, 3> bmin = { 0, 0, 0 };
	std::array<float, 3> bmax = { 0, 0, 0 };

	Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity();

	float modelScale = 1;
	Eigen::Vector3f modelShift = { 0, 0, 0 };

	std::array<std::vector<std::string>, MAX_TO_TRACK> texturesPaths;
	std::array<std::vector<GLuint>, MAX_TO_TRACK> texturesIDs;

	std::array<std::array<BoneTransformation, BONES_COUNT>, MAX_TO_TRACK> boneTransformations;

	FxWidgetFaceMask();

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;

	void onInputFrameResized();

	bool load() override;
	void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	void drawBones(cv::Mat frame, float scaleBones, Eigen::Vector2f moveBones);

	bool reloadOBJ();
};