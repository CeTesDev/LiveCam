#pragma once

#include "boost/filesystem.hpp"

#include <Models/FilterUiModel.h>

#include "FxCore.h"
#include <Widgets/FxWidget3D.h>

class FxTemplate : public FxCore
{
public:
	FxTemplate(FilterUiModel* externalInfo);
	FxTemplate::~FxTemplate();

	void load();
	void transform(FXModel& face, ExternalRenderParams &externalRenderParams);
};