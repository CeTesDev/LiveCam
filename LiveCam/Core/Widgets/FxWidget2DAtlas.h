#pragma once

#include <Widgets/FxWidget2DAlternate.h>

class FxWidget2DAtlas : public FxWidget2DAlternate
{
public:
	std::array<Eigen::Vector2f, MAX_TO_TRACK> atlasTextureScale;
	std::array<GLuint, MAX_TO_TRACK> animationAtlasID;

	FxWidget2DAtlas();

	std::array<std::vector<ResourceManager::AtlasLoadingResult::AtlasTexture>, MAX_TO_TRACK> atlasTextures;

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;
};
