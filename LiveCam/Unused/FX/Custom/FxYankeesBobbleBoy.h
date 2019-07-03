#pragma once

#include <fx\FxWidget3D.h>

class FxYankeesBobbleBoy : public FX
{
public:
	FxYankeesBobbleBoy();
	~FxYankeesBobbleBoy();

	int maxMovementBoy = 9;
	void animateNextStep(size_t i);

	std::array<int, FaceTracker::MAX_TO_TRACK> animationCounter;
	std::array<bool, FaceTracker::MAX_TO_TRACK> animationDirection;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;

}; 