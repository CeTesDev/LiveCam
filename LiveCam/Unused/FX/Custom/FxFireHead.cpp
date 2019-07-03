#include <fx/FxFireHead.h>

FxFireHead::FxFireHead()
{

	loadFromJSON("./assets/fx/miamiHeat/miami_heat_facepaint_with_fire.json");

	fire = (FxWidget2DAtlas*)models[2].get();
}

FxFireHead::~FxFireHead()
{
}

void FxFireHead::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	FX::transform(face, externalRenderParams);

	float angle = fire->renderParams.rollSmoother[face.pointId].smoothValue / 180 * M_PI - M_PI_2;
	float sine = sinf(angle);
	float cosine = cosf(angle);

	fire->animationOffset[face.pointId] = { -1.5f * cosine, 2 * sine, -0.6f };
	fire->animationScale[face.pointId][0] = 7 * (1 + 0.4f * abs(cosine));
}