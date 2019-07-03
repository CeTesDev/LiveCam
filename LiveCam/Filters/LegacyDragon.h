#pragma once

#include <Widgets/FxWidget3D.h>

#define TOTAL_NUMBER_OF_IMAGES 89

class LegacyDragon : public FX
{
public:
	LegacyDragon();
	~LegacyDragon();

	void onInputFrameResized() override;

	std::array<ValueSmoother, MAX_TO_TRACK> lipsDistanceSmoother;

	void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams) override;
};