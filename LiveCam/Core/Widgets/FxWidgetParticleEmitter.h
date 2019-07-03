#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsModel.h>
#include <FX.h>

class FxWidgetParticleEmitter : public GraphicsModel
{
public:
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	bool backClipping;
	float Xclip;
	float Yclip;
	float Zclip;

	std::array<float, 3> bmin = { 0, 0, 0 };
	std::array<float, 3> bmax = { 0, 0, 0 };

	Eigen::Matrix3f extraRotateMatrix = Eigen::Matrix3f::Identity();

	float modelScale = 1;
	Eigen::Vector3f modelShift = { 0, 0, 0 };

	std::array<ValueSmoother, MAX_TO_TRACK> pivotX;
	std::array<ValueSmoother, MAX_TO_TRACK> pivotY;
	Eigen::Vector3d pivotOffset;

	FxWidgetParticleEmitter();

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	boost::property_tree::ptree getPTree(ExtraModelData &data) override;

	void onInputFrameResized();

	bool load() override;
	void transform(TrackingTarget& fxModel, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& fxModel, ExternalRenderParams &externalRenderParams) override;
	void unload() override;
};
