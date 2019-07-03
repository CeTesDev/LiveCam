#pragma once

#include <Tracking/TrackingTarget.h>
#include <Graphics/GraphicsModel.h>

struct FaceDistortionParams
{
	friend class FaceDistortion;
	friend class GraphicsFaceDistortionModel;

	std::string iconImagePath;

	FaceDistortionParams();

	void applySuit(boost::property_tree::ptree& suit, bool loadTexturesImmediately);
	void load();

protected:
};

class FaceDistortion
{
	friend class FxWidgetFaceDistortion;

protected:
	GLuint vbo = 0;

	GLuint texCoordVBO = 0;

public:
	GLuint textureid = 0;

public:
	void load();
	void transform(ImagePoints &points, FaceDistortionParams &params);
	void draw(FaceDistortionParams &params, std::shared_ptr<cwc::glShader> shader);
	void unload();
};

class GraphicsFaceDistortionModel : public GraphicsModel
{
public:
	FaceDistortion facedistortion;
	std::array<FaceDistortionParams, MAX_TO_TRACK> faceParams;

	GraphicsFaceDistortionModel();

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

protected:
	GLuint textureid = 0;
};
