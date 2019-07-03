#pragma once

#include <map>
#include <string>

#include <Common/CommonClasses.h>
#include "FilterUiModel.h"
#include "FilterModel.h"
#include "FX.h"

class FilterManager
{
public:

	typedef std::function<std::shared_ptr<FX>()> StaticCreator;

	std::map<int, StaticCreator> associations;
	std::map<int, std::shared_ptr<FX>> filters;

	std::shared_ptr<FX> currentFilter;
	int currentFilterID;

	int newTemplatedFilterID = -1;

	FilterManager();
	~FilterManager();

	std::shared_ptr<FX> getCurrentFilter();
	int getCurrentFilterID();

	std::shared_ptr<FX> findFilterById(int filterID);

	void setCurrentFilterWithID(int filterID);

	void addFilter(int filterID);

	int addFilter(FilterUiModel* externalInfo);

	int addFilter(std::shared_ptr<FX> filter);

	void loadExternFilters(FilterUiModel& externalInfo, std::vector<FilterUiModel>& otherInfo);
};
