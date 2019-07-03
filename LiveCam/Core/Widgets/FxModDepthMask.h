#pragma once

#include <Widgets/FxWidget3D.h>
#include <Common/CommonClasses.h>

#define _USE_MATH_DEFINES
#include <math.h>

class DepthMask : public FxWidget3D
{
public:
	static const std::string TYPE_NAME;
	std::string getTypeName() override;

	const static size_t FAN_LIMIT = 23;;
	static const std::array<int, FAN_LIMIT> fan_modifiers;

	DepthMask();
	DepthMask(float zOffset, bool draw2Dmask, bool draw3Dmask);
	~DepthMask();

	void onInputFrameResized() override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;
	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately) override;

	bool load() override;
	void transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	float zOffset;
	bool draw2Dmask;
	bool draw3Dmask;

protected:
	std::shared_ptr<cwc::glShader> shader2D;
	std::shared_ptr<cwc::glShader> shader3D;
	ModelRenderParams params2D;
	ModelRenderParams params3D;
	GLuint mask2dVBO;
};