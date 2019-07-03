#pragma once

#include <fx\FxWidget3D.h>
#include "models/FxWidgetSprite.h"

class FxTeslaHelmetOKC : public FX
{
public:
	std::shared_ptr<ObjectRenderParams> helmetRenderParams;
	std::shared_ptr<FxWidget3D> helmet;

	FxTeslaHelmetOKC();
	~FxTeslaHelmetOKC();

	void load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	FxWidgetSprite boltSprite;

	static const int PHI_COUNT = 3;
	static const int ALPHA_COUNT = 12;
	static const int MAX_RELOADING_TIME = 50;

	std::array<std::array<int, PHI_COUNT>, ALPHA_COUNT> animationStart;
	std::array<std::array<int, PHI_COUNT>, ALPHA_COUNT> animationIndex;
};