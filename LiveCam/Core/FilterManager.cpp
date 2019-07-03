#include "FilterManager.h"
#include <Filters/_Constants.h>
#include <Filters/BlackHoleSun.h>

FilterManager::FilterManager()
{
	associations[BIG_EYES_EFFECT] = &FX::create<BlackHoleSun>;
}

FilterManager::~FilterManager()
{
}
std::shared_ptr<FX> FilterManager::getCurrentFilter()
{
	return currentFilter;
}

int FilterManager::getCurrentFilterID()
{
	return currentFilterID;
}

void FilterManager::setCurrentFilterWithID(int filterID)
{
	auto newFilter = filters[filterID];
	if (newFilter != nullptr)
	{
		currentFilter = newFilter;
		currentFilterID = filterID;
	}
}

std::shared_ptr<FX> FilterManager::findFilterById(int filterID)
{
	return filters[filterID];
}

void FilterManager::addFilter(int filterID)
{
	if (filterID < 1)
	{
		return;
	}

	auto creator = associations[filterID];
	if (creator != nullptr)
	{
		auto filter = filters[filterID] = creator();
		if (currentFilter == nullptr)
		{
			currentFilter = filter;
			currentFilterID = filterID;
		}
	}
}

int FilterManager::addFilter(FilterUiModel* externalInfo)
{
	std::shared_ptr<FX> ptr;
	
	try
	{
		ptr = std::make_shared<FilterModel>(externalInfo);
	}
	catch (std::exception&)
	{
		return 0;
	}

	filters[newTemplatedFilterID] = ptr;
	if (currentFilter == nullptr)
	{
		currentFilter = filters[newTemplatedFilterID];
		currentFilterID = newTemplatedFilterID;
	}

	externalInfo->setId(newTemplatedFilterID);
	return newTemplatedFilterID--;
}

int FilterManager::addFilter(std::shared_ptr<FX> filter)
{
	filters[newTemplatedFilterID] = filter;
	if (currentFilter == nullptr)
	{
		currentFilter = filters[newTemplatedFilterID];
		currentFilterID = newTemplatedFilterID;
	}
	return newTemplatedFilterID--;
}

void FilterManager::loadExternFilters(FilterUiModel& externalInfo, std::vector<FilterUiModel>& otherInfo)
{
	if (externalInfo.getJSONsource().empty())
	{
		return;
	}

	auto filter = findFilterById(externalInfo.getId());

	boost::property_tree::ptree source;
	boost::property_tree::json_parser::read_json(externalInfo.getJSONsource(), source);

	auto externList = source.get_child_optional("extern");
	if (externList)
	{
		for (auto &externFilter : externList.get())
		{
			auto value = externFilter.second.get_value<std::string>();

			int id = 0;
			try
			{
				id = std::stoi(value);
				filter->externFilters.push_back(associations[id]());
				continue;
			}
			catch (...) {}

			for (auto other : otherInfo)
			{
				if (other.getJSONsource() == value)
				{
					auto externFilter = filters[other.getId()];
					if (externFilter->externFilters.size() > 0)
					{
						throw new std::exception("Included filter can not include any other filter!");
					}
					filter->externFilters.push_back(filters[other.getId()]);
					continue;
				}
			}

			FilterUiModel externalInfo;
			externalInfo.setJSONsource(value);
			auto externFilter = std::make_shared<FX>(FilterModel(&externalInfo));

			filter->externFilters.push_back(externFilter);
		}
	}

	for (auto &module : filter->filterModules)
	{
		if (module.externFilterID == -1)
		{
			continue;
		}

		if (module.externModuleID > -1)
		{
			auto& externModule = filter->externFilters[module.externFilterID]->filterModules[module.externModuleID];
			module.icon = externModule.icon;
			module.iconPath = externModule.iconPath;
			module.suits = externModule.suits;
		}
	}
}