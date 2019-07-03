#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <3D/GraphicsModel.h>

class GraphicsHand2DModel : public GraphicsModel
{
protected:
	std::array<Eigen::Vector2f, ObjectTracker::MAX_TO_TRACK> textureSize;

public:
	GraphicsHand2DModel();

	std::array<std::string, ObjectTracker::MAX_TO_TRACK> handsTextures;
	std::array<GLuint, ObjectTracker::MAX_TO_TRACK> handsTexturesIDs;

	std::array<Eigen::Vector3f, ObjectTracker::MAX_TO_TRACK> handsOffset;
	std::array<Eigen::Vector2f, ObjectTracker::MAX_TO_TRACK> handsScale;
	std::array<Eigen::Vector2f, ObjectTracker::MAX_TO_TRACK> handsPivot; // texture rotation pivot in texture coord

	std::array<bool, ObjectTracker::MAX_TO_TRACK> match3D;

	std::array<int, ObjectTracker::MAX_TO_TRACK> symmetricState;
	// -1: default image is not symmetric & looks at left; 0: is symmetric, no X mirroring needed; 1: looks at right

	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	void applySuit(boost::property_tree::ptree& suit, size_t faceIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;

	boost::property_tree::ptree getPTree(ExtraModelData &data) override;
	void prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData) override;

	bool load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;

	std::array<bool, ObjectTracker::MAX_TO_TRACK> handRotation;
	std::array<bool, ObjectTracker::MAX_TO_TRACK> handRolling;


	// for LiveCam Editor
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;

};

class Hand2DModel : public GraphicsHand2DModel
{
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;

};
