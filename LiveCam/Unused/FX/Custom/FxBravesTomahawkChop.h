#pragma once

#include <fx\FxWidget3D.h>

class FxBravesTomahawkChop: public FX
{
public:
	FxBravesTomahawkChop();
	~FxBravesTomahawkChop();

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	
private:

	std::array<ValueSmoother, FaceTracker::MAX_TO_TRACK> lipsDistanceSmoother;

	void animateNextStep(size_t i);

	std::array<int, FaceTracker::MAX_TO_TRACK> animationCounter;
	std::array<bool, FaceTracker::MAX_TO_TRACK> animationDirection;
}; 