#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsModel.h>
#include <FX.h>

class FxWidget2D : public GraphicsModel
{
protected:
	std::array<Eigen::Vector2f, MAX_TO_TRACK> textureSize;

public:
	FxWidget2D();

	std::array<std::string, MAX_TO_TRACK> emojisTextures;
	std::array<GLuint, MAX_TO_TRACK> emojisTexturesIDs;

	std::array<Eigen::Vector3f, MAX_TO_TRACK> emojisOffset;
	std::array<Eigen::Vector2f, MAX_TO_TRACK> emojisScale;
	std::array<Eigen::Vector2f, MAX_TO_TRACK> emojisPivot;

	std::array<bool, MAX_TO_TRACK> match3D;

	std::array<int, MAX_TO_TRACK> symmetricState;

	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;

	boost::property_tree::ptree getPTree(ExtraModelData &data) override;
	void prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;

	std::array<bool, MAX_TO_TRACK> emojiRotation;
	std::array<bool, MAX_TO_TRACK> emojiRolling;
	std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams() override;
	void setDefaultObjectRenderParams(ObjectRenderParams& params) override;

};

class Suit2DModel : public FxWidget2D
{
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;

};