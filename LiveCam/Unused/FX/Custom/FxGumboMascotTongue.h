#pragma once

#include <fx\FxWidget3D.h>
#include "models/FxWidgetSprite.h"

#include <boost/lexical_cast.hpp>
#include <boost/chrono/chrono_io.hpp>

class FxGumboMascotTongue : public FX
{
private:
	std::shared_ptr<cwc::glShader> shader3D;
	std::shared_ptr<cwc::glShader> shader3Dtongue;

	float tongueX[15] = { 0.2f, 0.2f, 0.19f, 0.16f, 0.13f, 0.1f, 0.1f, 0.1f, 0.12f, 0.13f, 0.16f, 0.19f, 0.2f, 0.2f, 0.2f };
	float tongueY[15] = { 0.2f, 0.2f, 0.19f, 0.16f, 0.13f, 0.1f, 0.1f, 0.1f, 0.12f, 0.13f, 0.16f, 0.19f, 0.2f, 0.2f, 0.2f };
	float tongueZ[15] = { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.1f, 1.1f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f };

	std::chrono::milliseconds timeStart;
	std::chrono::milliseconds timeEnd;
	long long int timeInterval;
	int timeCount = 0;
	int dt = 25;
	int countFrame = 0;

public:
	FxGumboMascotTongue();
	~FxGumboMascotTongue();

	std::shared_ptr<FxWidget3D> dogTongueModel;
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
	void draw(FXModel& face, ExternalRenderParams &externalRenderParams) override;

};