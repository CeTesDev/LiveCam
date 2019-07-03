#include <fx/FxRedsFaceSwap.h>
#include <utils/DrawUtil.h>
#include <iostream>
#include "utils/Resolutions.h"

extern FaceTracker tracker;

FxRedsFaceSwap::FxRedsFaceSwap()
{
	loadFromJSON("./assets/fx/reds/reds_face_swap_modules.json");
}

FxRedsFaceSwap::~FxRedsFaceSwap()
{
}
