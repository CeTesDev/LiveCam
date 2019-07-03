#pragma once

#include <fx\FxWidget3D.h>

class FxCelticsBobbleHead: public FX
{
public:
	FxCelticsBobbleHead();
	~FxCelticsBobbleHead();

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	
private:
	void animateNextStep(size_t i);

	std::array<int, FaceTracker::MAX_TO_TRACK> animationCounter;
	std::array<bool, FaceTracker::MAX_TO_TRACK> animationDirection;
}; 