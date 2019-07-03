#pragma once

#include <fx\FxWidget3D.h>

#define TOTAL_NUMBER_OF_IMAGES 89

class FxDragon : public FX
{
public:
	FxDragon();
	~FxDragon();

	void onInputFrameResized() override;

	std::array<ValueSmoother, FaceTracker::MAX_TO_TRACK> lipsDistanceSmoother;

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
};