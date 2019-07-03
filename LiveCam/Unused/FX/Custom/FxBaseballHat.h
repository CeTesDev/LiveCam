#include <fx\FxWidget3D.h>

#pragma once

class FxBaseballHat : public FX
{
public:
	FxBaseballHat();
	~FxBaseballHat();

	void load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;

private:
	std::shared_ptr<FxWidget3D> headModel;

	std::shared_ptr<cwc::glShader> shader = NULL;

	GLuint hatTexture;
	GLuint extraTexture;
	GLuint normalMap;
};