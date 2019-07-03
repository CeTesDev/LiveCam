#ifndef LIVECAM_WIDGETS_CONSTANTS_H
#define LIVECAM_WIDGETS_CONSTANTS_H

#include <Graphics/GraphicsModel.h>

#include "FxWidget3D.h"
#include "FxWidget3DAnimated.h"
#include "FxWidget2D.h"
#include "FxWidget2DAnimated.h"
#include "FxWidget2DAlternate.h"
#include "FxWidget2DAtlas.h"
#include "FxModDepthMask.h"
#include "FxModAlphaMask.h"
#include "FxWidgetFacePaint.h"
#include "FxWidgetFaceSwap.h"
#include "FxWidgetFaceMask.h"
#include "FxWidgetParticleEmitter.h"

//  Global Constants for Widgets values
static std::map<std::string, std::function<std::shared_ptr<GraphicsModel>()>> MODEL_ASSOCIATIONS =
{
	{ "3DModel", &GraphicsModel::create<FxWidget3D> },
	{ "3DAnimatedModel", &GraphicsModel::create<FxWidget3DAnimated> },
	{ "2DModel", &GraphicsModel::create<FxWidget2D> },
	{ "2DAnimatedModel", &GraphicsModel::create<FxWidget2DAnimated> },
	{ "LipsJoint3DModel", &GraphicsModel::create<LipsJoint3DModel> },
	{ "LipsBottomJoint3DModel", &GraphicsModel::create<LipsBottomJoint3DModel> },
	{ "LeftBrow3DModel", &GraphicsModel::create<LeftBrow3DModel> },
	{ "RightBrow3DModel", &GraphicsModel::create<RightBrow3DModel> },
	{ "Suit2DModel", &GraphicsModel::create<Suit2DModel> },
	{ "FacePaintModel", &GraphicsModel::create<FxWidgetFacePaint> },
	{ "DepthMask", &GraphicsModel::create<DepthMask> },
	{ "AlphaMask", &GraphicsModel::create<AlphaMask> },
	{ "FaceSwapModel", &GraphicsModel::create<FxWidgetFaceSwap> },
	{ "Static3DModel", &GraphicsModel::create<Static3DModel> },
	{ "Alternate2DModel", &GraphicsModel::create<FxWidget2DAlternate> },
	{ "Atlas2DModel", &GraphicsModel::create<FxWidget2DAtlas> },
	{ "FaceMaskModel", &GraphicsModel::create<FxWidgetFaceMask> },
	{ "ParticleEmitter", &GraphicsModel::create<FxWidgetParticleEmitter> }
};


#endif
