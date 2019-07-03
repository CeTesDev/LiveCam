#pragma once

#include <fx\FxWidget3D.h>
#include "models/FxWidgetSprite.h"

class FxLightningHair : public FX
{
public:
	std::shared_ptr<ObjectRenderParams> helmetRenderParams;
	std::shared_ptr<FxWidget3D> helmet;

	FxLightningHair();
	~FxLightningHair();

	std::array<ValueSmoother, FaceTracker::MAX_TO_TRACK> lipsDistanceSmoother;

	void onInputFrameResized() override;

	void load() override;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void unload() override;

	FxWidgetSprite boltSprite;

	static const int CHARGES_COUNT = 30;
	static const int MAX_RELOADING_TIME = 50;

	std::array<int, CHARGES_COUNT> animationStart;
	std::array<int, CHARGES_COUNT> animationIndex;
	int lipsAnimationIndex;

	std::array<std::pair<float, float>, CHARGES_COUNT> chargesCoords;
};