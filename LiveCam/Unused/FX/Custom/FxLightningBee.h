#pragma once

#include <fx\FxWidget3D.h>
#include "models/FxWidgetSprite.h"

class FxLightningBee : public FX
{
public:
	std::shared_ptr<ObjectRenderParams> helmetRenderParams;
	std::shared_ptr<FxWidget3D> helmet;

	FxLightningBee();
	~FxLightningBee();

	void load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	FxWidgetSprite boltSprite;

	int animationIndex;
};