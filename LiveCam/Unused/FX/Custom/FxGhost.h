#pragma once

#include <fx\FxWidget3D.h>

class FxGhost : public FX
{
public:
	FxGhost();
	~FxGhost();

	std::shared_ptr<cwc::glShader> ghostShader;

	std::shared_ptr<FxWidget3D> ghost;
	float wavesOffset;
	GLuint wavesTex;

	float shineFactor;
	bool shineIncreasing;

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;
}; 