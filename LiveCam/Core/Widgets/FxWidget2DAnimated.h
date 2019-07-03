#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsModel.h>
#include <Common/PointCalcUtil.h>

enum AnimationState
{
	CLOSED,
	OPENING,
	OPENED,
	CLOSING
};

class FxWidget2DAnimated : public GraphicsModel
{
protected:
	std::array<Eigen::Vector2f, MAX_TO_TRACK> animationSize;
public:
	FxWidget2DAnimated();

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	virtual void animate(TrackingTarget& face);

	std::array<bool, MAX_TO_TRACK> animationRotation;
	std::array<bool, MAX_TO_TRACK> animationRolling;

	std::array<int, MAX_TO_TRACK> symmetricState;
	std::array<std::string, MAX_TO_TRACK> animationPaths;

	std::array<Eigen::Vector3f, MAX_TO_TRACK> animationOffset;
	std::array<Eigen::Vector2f, MAX_TO_TRACK> animationScale;
	std::array<Eigen::Vector2f, MAX_TO_TRACK> animationPivot;

	std::array<bool, MAX_TO_TRACK> match3D;

	std::array<size_t, MAX_TO_TRACK> animationOpened;
	std::array<size_t, MAX_TO_TRACK> animationLength;
	std::array<float, MAX_TO_TRACK> animationSpeed;

	std::array<std::vector<GLuint>, MAX_TO_TRACK> animationFramesIDs;
	std::array<float, MAX_TO_TRACK> animationIndexes;
	std::array<AnimationState, MAX_TO_TRACK> animationState;
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;
};