#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsModel.h>
#include <FX.h>

using namespace Eigen;

class FxWidget3DFBX : public GraphicsModel
{
public:
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	bool backClipping;
	float Xclip;
	float Yclip;
	float Zclip;
	std::vector<std::string> materialList;
	std::vector<std::string> modelList;

	std::string modelPath;
	std::string materialPath;

	std::array<float, 3> bmin = { 0, 0, 0 };
	std::array<float, 3> bmax = { 0, 0, 0 };

	Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity();

	float modelScale = 1;
	Eigen::Vector3f modelShift = { 0, 0, 0 };

	std::array<std::vector<std::string>, ObjectTracker::MAX_TO_TRACK> texturesPaths;
	std::array<std::vector<GLuint>, ObjectTracker::MAX_TO_TRACK> texturesIDs;

	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> pivotX;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> pivotY;
	Eigen::Vector3d pivotOffset;

	FxWidget3DFBX();

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	boost::property_tree::ptree getPTree(ExtraModelData &data) override;

	void prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData) override;

	void onInputFrameResized();

	bool load() override;
	void transform(FXModel& fxModel, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& fxModel, ExternalRenderParams &externalRenderParams) override;
	void unload() override;
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;

	bool reloadOBJ();
	void sortMeshesInZAscending();
};

#include "FxModDepthMask.h"
