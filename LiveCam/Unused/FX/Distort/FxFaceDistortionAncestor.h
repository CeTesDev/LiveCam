#pragma once

#include "models/FXModel.h"
#include <3D/GraphicsModel.h>

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
	friend class FxDistortionAncestor;

protected:
	const static size_t LINE_STRIP_COUNT = 16;

	const static size_t CHIN_STRIP_COUNT = 9;
	const static size_t CHEEK_FAN_COUNT = 10;
	const static size_t BROW_STRIP_COUNT = 10;
	const static size_t NOSE_FAN_COUNT = 8;
	const static size_t UNDER_NOSE_FAN_COUNT = 6;
	const static size_t FOREHEAD_STRIP_COUNT = 9;

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
	std::array<FaceDistortionParams, ObjectTracker::MAX_TO_TRACK> faceParams;

	GraphicsFaceDistortionModel();

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

protected:
	GLuint textureid = 0;
};