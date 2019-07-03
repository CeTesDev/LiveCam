#include <FX.h>

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <GL/glut.h>
// #include <fbxsdk.h>

#include <Widgets/_Constants.h>
#include <LiveCamWindow.h>

extern Detector detector;

ParticleManager particleManager;

FilterModule::FilterModule()
{
	moduleID = -1;
	externFilterID = -1;
	externModuleID = -1;
}

FilterModule::FilterModule(boost::property_tree::ptree &moduleTree, size_t moduleID, const fs::path &resourceRoot)
{
	externFilterID = moduleTree.get<int>("externFilterID", -1);
	externModuleID = moduleTree.get<int>("externModuleID", -1);

	this->moduleID = moduleID;

	iconPath = moduleTree.get<std::string>("iconPath", "");

	icon.addFile(QString(fs::path(resourceRoot / iconPath).string().c_str()), QSize(40, 40), QIcon::Normal);
	icon.addFile(QString(fs::path(resourceRoot / iconPath).string().c_str()), QSize(40, 40), QIcon::Disabled);

	if (externFilterID > -1)
	{
		return;
	}

	auto suitsTree = moduleTree.get_child_optional("suits");
	if (suitsTree)
	{
		for (auto &suit : suitsTree.get())
		{
			suits.push_back(suit.second);
		}
	}
}

boost::property_tree::ptree FilterModule::getPTree(ExtraModuleData &data, std::vector<std::shared_ptr<GraphicsModel>> &models,
	TCommonRenderParams& commonRenderParams, int moduleIndex)
{
	boost::property_tree::ptree result;

	fs::path path(data.moduleIconPath);

	result.put("iconPath", path.filename().string());

	boost::property_tree::ptree suitsTree;

	int i = 0;
	auto model = models.begin();
	for (auto &suitRecord : suits)
	{
		if (i++ == moduleIndex)
		{
			models[moduleIndex]->prepareSuitForJSON(suits[moduleIndex], data.models[0]);
		}
		suitsTree.push_back(make_pair("", suitRecord));
	}

	result.put_child("suits", suitsTree);

	if (!data.moduleIconPath.empty())
	{
		auto copiedFile = fs::path(data.root->filterFolder / path.filename());
		if (!equivalent(path.string(), copiedFile))
		{
			fs::copy_file(path.string(), copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	return result;
}

FX::FX()
{ 
	resetStable();
	resourcesRoot = "";
	trackingType = "FACE";
	editable = false;
	useFrameLightParams = false;
	useFaceLightParams = false;
}

FX::~FX()
{ 
}

void FX::outerTransform(TrackingTarget& target, ExternalRenderParams externalRenderParams)
{
	this->transform(target, externalRenderParams);
}

void FX::outerDraw(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	this->draw(target, externalRenderParams);
}

bool FX::detectInside(std::vector<cv::Rect>& opencvFaces, cv::Rect rect)
{

	for (size_t i = 0; i < opencvFaces.size(); i++)
	{
		if ((rect.x+ rect.width/2 > opencvFaces[i].x + opencvFaces[i].width/4) &&
			(rect.x + rect.width / 2 < opencvFaces[i].x + (opencvFaces[i].width * 3) / 4) &&
			(rect.y + rect.height / 2 > opencvFaces[i].y + opencvFaces[i].height / 4) &&
			(rect.y + rect.height / 2 < opencvFaces[i].y + (opencvFaces[i].height * 3) / 4))
		{
			return true;
		}
	}

	return false;
}

void FX::resetStable()
{
	for (int i = 0; i < MAX_TO_TRACK; i++)
	{
		becameStable[i] = false;
	}
}

void FX::applyModule(FilterModule *module, size_t targetIndex, bool loadTexturesImmediately)
{
	FX* filterPtr = nullptr;
	FilterModule* modulePtr = nullptr;

	if (module->externFilterID == -1)
	{
		modulePtr = module;
		filterPtr = this;
	}
	else
	{
		filterPtr = externFilters[module->externFilterID].get();
		if (module->externModuleID > -1)
		{
			modulePtr = &filterPtr->filterModules[module->externModuleID];
		}
	}

	std::vector<FX*> otherFilters;

	if (module->externFilterID != -1)
	{
		otherFilters.push_back(this);
	}

	int index = 0;
	for (auto other : externFilters)
	{
		if (module->externFilterID != index)
		{
			otherFilters.push_back(other.get());
		}
		++index;
	}

	boost::property_tree::ptree disablingSuit;
	disablingSuit.put("visible", "false");

	if (modulePtr)
	{
		std::vector<bool> appliedSuit(filterPtr->models.size(), false);

		int modelID = 0;
		int suitID = 0;
		for (auto suit : modulePtr->suits)
		{
			while (!filterPtr->models[modelID]->canSwapSuit)
			{
				appliedSuit[modelID] = true;
				++modelID;
			}

			int modelID_local = modelID;

			auto modelIDrecord = suit.get_optional<int>("modelID");
			if (modelIDrecord)
			{
				modelID_local = modelIDrecord.get();
			}

			++suitID;
			++modelID;
			
			auto model = filterPtr->models[modelID_local];

			appliedSuit[modelID_local] = true;

			if (model->canSwapSuit)
			{
				/*auto visible_x = suit.get<bool>("visible", true);
				auto alphaMask_x = suit.get<bool>("useAlphaMask", false);
				auto animateTexture_x = suit.get<bool>("animateTextures", false);*/

				model->applySuit(suit, targetIndex, filterPtr->commonRenderParams, loadTexturesImmediately);
			}
		}

		for (modelID = 0; modelID < filterPtr->models.size(); ++modelID)
		{
			if (!appliedSuit[modelID])
			{
				filterPtr->models[modelID]->applySuit(disablingSuit, targetIndex, filterPtr->commonRenderParams, loadTexturesImmediately);
			}
		}
	}
	else
	{
		boost::property_tree::ptree enablingSuit;
		enablingSuit.put("visible", "true");
		for (auto model : filterPtr->models)
		{
			if (model->canSwapSuit)
			{
				model->applySuit(enablingSuit, targetIndex, filterPtr->commonRenderParams, loadTexturesImmediately);
			}
		}
	}

	for (auto other : otherFilters)
	{
		for (auto model : other->models)
		{
			if (model->canSwapSuit)
			{
				model->applySuit(disablingSuit, targetIndex, other->commonRenderParams, loadTexturesImmediately);
			}
		}
	}
}

void FX::onInputFrameResized()
{
	for (auto model : models)
	{
		model->onInputFrameResized();
	}

	for (auto filter : externFilters)
	{
		for (auto model : filter->models)
		{
			model->onInputFrameResized();
		}
	}
}
void FX::updateModules(int moduleId, int targetIndex)
{
	applyModule(&filterModules[moduleId], targetIndex, false);
}
void FX::load()
{
	if (!cubemapTextures[0].empty())
	{
		cubemapID = resourceManager.loadCubemap(cubemapTextures);
	}

	srand(time(NULL));
	
	if (filterModules.size() > 0)
	{
		for (int i = 0; i < MAX_TO_TRACK; ++i)
		{
			initialModulesRandomize
				? applyModule(&filterModules[rand() % filterModules.size()], i, false)
				: applyModule(&filterModules[initialModules[i]], i, false);
		}

	}
	
	useAlphaMask = false;

	for (auto model : models)
	{
		model->load();
		if (!useAlphaMask && model->getTypeName() == "AlphaMask")
		{
			useAlphaMask = true;
		}
	}

	for (auto filter : externFilters)
	{
		for (auto model : filter->models)
		{
			model->load();
		}
	}

	for (auto effect : particleEffects)
	{
		effect->load();
		effect->startSpawn();
	}

	detector.setTargetRestriction(maxOnScreen);
	detector.setTrackingType(trackingType);
}

void FX::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	for (auto model : models)
	{
		if (model->visible[target.pointId])
		{
			model->transform(target, externalRenderParams);
		}
	}

	for (auto filter : externFilters)
	{
		filter->transform(target, externalRenderParams);
	}

	for (auto effect : particleEffects)
	{
		ParticleUpdateParams params;

		params.outer_transformation = Matrix4f::Identity();
		params.effect_rotation = Matrix3f::Identity();
		params.effect_scaling = Affine3f(Scaling(effect->getEffectWidth() / Resolutions::OUTPUT_WIDTH)).matrix();
		params.enable_effect_rotation = false;
		params.globalGravity = { 0, 0, 0 };
		params.enable_effect_gravity = true;
		params.enable_global_gravity = false;

		Matrix3f identity3 = Matrix3f::Identity();
		effect->update(externalRenderParams.timeShift, params);
	}
}

void FX::draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	if (cubemapID != 0)
	{
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapID);
	}

	for (auto model : models)
	{
		if (!model->visible[target.pointId])
		{
			continue;
		}
		if (model->useAlphaMask[target.pointId])
		{
			frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::FILTER_MASK);
		}
		else
		{
			frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::COLOR_1);
		}

		model->draw(target, externalRenderParams);
		model->postprocess(target);
	}

	for (auto filter : externFilters)
	{
		filter->draw(target, externalRenderParams);
	}
}

void FX::drawParticles(Matrix4f& projection)
{
	for (auto effect : particleEffects)
	{
		effect->draw(projection);
	}
}

void FX::unload()
{
	for (auto model : models)
	{
		model->unload();
	}

	for (auto filter : externFilters)
	{
		for (auto model : filter->models)
		{
			model->unload();
		}
	}

	for (auto effect : particleEffects)
	{
		effect->freeze();
		effect->unload();
	}
}

void FX::outerSetCMID(GLuint cmId)
{
	for (auto model : models)
	{
		model->setCMID(cmId);
	}
}

void FX::loadFromJSON(const fs::path& path, FilterUiModel* externalInfo)
{
	boost::property_tree::ptree filterTree;
	boost::property_tree::read_json(path.string(), filterTree);

	useFaceLightParams = filterTree.get<bool>("use face light params", false);
	useFrameLightParams = filterTree.get<bool>("use frame light params", false);

	bool JSONroot = filterTree.get<bool>("JSON root", false);
	if (JSONroot)
	{
		resourcesRoot = path.parent_path();
	}

	auto iconPath = filterTree.get_optional<std::string>("iconPath");
	if (iconPath && externalInfo)
	{
		externalInfo->setIcon(iconPath.get());
	}

	auto title = filterTree.get_optional<std::string>("title");
	if (title && externalInfo)
	{
		externalInfo->setTitle(title.get());
	}

	auto skyboxRecord = filterTree.get_child_optional("skybox");
	if (skyboxRecord)
	{
		cubemapTextures = JSONVectorReader::readVector<std::string, 6>(skyboxRecord.get());
	}

	auto modelsList = filterTree.get_child_optional("models");
	if (modelsList)
	{
		editable = true;

		for (auto &modelRecord : modelsList.get())
		{
			std::string modelType = modelRecord.second.get<std::string>("type", "");

			auto model = MODEL_ASSOCIATIONS[modelType]();

			model->loadFromJSON(modelRecord.second);

			model->useHardCodedUniforms = false;

			models.push_back(model);
		}
	}

	auto renderParamsList = filterTree.get_child_optional("renderParams");
	if (renderParamsList)
	{
		for (auto &renderParams : renderParamsList.get())
		{
			commonRenderParams.push_back(make_shared<ObjectRenderParams>(renderParams.second, resourcesRoot));
		}
	}

	auto modulesList = filterTree.get_child_optional("modules");
	if (modulesList)
	{
		size_t ID = 0;
		for (auto &module : modulesList.get())
		{
			filterModules.push_back(FilterModule(module.second, ID++, resourcesRoot));
		}
	}

	auto initialModulesList = filterTree.get_child_optional("initialModules");

	if (initialModulesList)
	{
		initialModules = JSONVectorReader::readVector<size_t, MAX_TO_TRACK>(initialModulesList.get());

		int p = 0;
		for (int i = initialModulesList->size(); i < MAX_TO_TRACK; ++i)
		{
			initialModules[i] = initialModules[p];

			if (++p == initialModulesList->size())
			{
				p = 0;
			}
		}
	}
	else
	{
		initialModules.fill(0);
	}

	initialModulesRandomize = filterTree.get<bool>("initialModulesRandomize", true);

	auto particles = filterTree.get_child_optional("particles");
	if (particles)
	{
		for (auto &particleEffect : particles.get())
		{
			auto effect = particleManager.addEffect(particleEffect.second.get<std::string>("JSONsource", ""));
			if (effect)
			{
				particleEffects.push_back(effect);
			}

			auto coordsRecord = particleEffect.second.get_child_optional("coords");
			auto coords = coordsRecord ? JSONVectorReader::readVector3f(coordsRecord.get()) : Vector3f(0.5f, 0.5f, 0);
			coords -= Vector3f(0.5, 0.5, 0);

			effect->setCoords({ coords[0] * Resolutions::OUTPUT_WIDTH, coords[1] * Resolutions::OUTPUT_HEIGHT, coords[2] / 10000 });
			effect->setDirection({ 0, 0 });
			effect->setSpeed({ 0, 0, 0 });
			effect->setDepthTest(false);
		}
	}

	trackingType = filterTree.get<std::string>("trackingType", "FACE");
	maxOnScreen = filterTree.get<int>("maxOnScreen", 0);
}

void FX::saveToJSON(const fs::path& filterFolder, ExtraFilterData &data)
{
	boost::property_tree::ptree filterTree;

	data.filterFolder = filterFolder;

	filterTree.put("JSON root", "true");

	filterTree.put("use newest transform", "true");

	if (!data.title.empty())
	{
		filterTree.put("title", data.title);
	}

	if (!data.filterIconPath.empty())
	{
		filterTree.put("iconPath", fs::path(data.filterIconPath).filename().string());

		auto copiedFile = fs::path(filterFolder / fs::path(data.filterIconPath).filename());
		if (!equivalent(data.filterIconPath, copiedFile))
		{
			fs::copy_file(data.filterIconPath, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	if (!cubemapTextures[0].empty())
	{
		boost::property_tree::ptree skyboxTree;

		for (auto &texture : cubemapTextures)
		{
			boost::property_tree::ptree skyboxTreeChild;
			skyboxTreeChild.put("", texture);
			skyboxTree.push_back(make_pair("", skyboxTreeChild));
		}

		filterTree.put_child("skybox", skyboxTree);
	}

	boost::property_tree::ptree modelsTree;
	if (models.size() > 0)
	{
		auto extraModuleData = data.modules.begin();
		for (auto model : models)
		{
			modelsTree.push_back(std::make_pair("", (model->getPTree(extraModuleData++->models[0]))));
		}
	}

	boost::property_tree::ptree initialModulesTree;
	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		boost::property_tree::ptree initialModulesTreeChild;
		initialModulesTreeChild.put("", 0);
		initialModulesTree.push_back(make_pair("", initialModulesTreeChild));
	}

	data.renderParams.clear();
	commonRenderParams.clear();
	boost::property_tree::ptree modulesTree;
	if (filterModules.size() > 0)
	{
		auto extraModuleData = data.modules.begin();
		for (int index = 0; index < filterModules.size(); ++index)
		{
			modulesTree.push_back(std::make_pair("", (filterModules[index].getPTree(*extraModuleData++, models, commonRenderParams, index))));
		}
	}

	boost::property_tree::ptree renderParamsTree;
	if (commonRenderParams.size() > 0)
	{
		auto renderParamsData = data.renderParams.begin();
		for (auto renderParams : commonRenderParams)
		{
			renderParamsTree.push_back(std::make_pair("", (renderParams->getPTree(*renderParamsData++))));
		}
	}

	if (models.size() > 0)
	{
		filterTree.put_child("models", modelsTree);
	}

	if (commonRenderParams.size() > 0)
	{
		filterTree.put_child("renderParams", renderParamsTree);
	}

	if (filterModules.size() > 0)
	{
		filterTree.put_child("initialModules", initialModulesTree);
		filterTree.put_child("modules", modulesTree);
	}

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		applyModule(&filterModules[0], i, true);
	}

	boost::property_tree::write_json((filterFolder / "filter-config.json").string(), filterTree);
}

void FX::transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *mesh)
{
	for (auto model : models)
	{
		model->transformMesh(frame, targets, mesh);
	}

	for (auto filter : externFilters)
	{
		for (auto model : filter->models)
		{
			model->transformMesh(frame, targets, mesh);
		}
	}
}