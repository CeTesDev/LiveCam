#pragma once

#include "boost/filesystem.hpp"

#include "FilterUiModel.h"
#include "FX.h"
#include <Widgets/FxWidget3D.h>

class FilterModel : public FX
{
public:
	FilterModel(FilterUiModel* externalInfo);
	FilterModel::~FilterModel();

	void load();
	void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams);
};