#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <boost/optional/optional.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <QDebug.h>
#include <QIcon.h>

#include <Common/CommonClasses.h>
#include <Common/PointCalcUtil.h>
#include <Common/CVImageUtil.h>
#include <Tracking/_Constants.h>
#include <Tracking/TrackingTarget.h>
#include <FilterUiModel.h>
#include <ResourceManager.h>
#include <Particles/ParticleManager.h>
#include <Graphics/GraphicsModel.h>

namespace fs = boost::filesystem;

extern ResourceManager resourceManager;

extern ShaderManagerWrapper shaderManagerWrapper;

struct FilterModule
{
	FilterModule();
	FilterModule(boost::property_tree::ptree &moduleRecord, size_t ID, const fs::path &resourceRoot);

	boost::property_tree::ptree getPTree(ExtraModuleData &data, std::vector<std::shared_ptr<GraphicsModel>> &models,
		TCommonRenderParams& commonRenderParams, int moduleIndex);

	std::vector<boost::property_tree::ptree> suits;

	size_t moduleID;
	std::string iconPath;
	QIcon icon;

	int externFilterID;
	int externModuleID;
};

class FX
{
public:
	FX();

	virtual ~FX();

	template <class T>
	static std::shared_ptr<FX> create()
	{
		return std::make_shared<T>();
	}

	bool useFrameLightParams;
	bool useFaceLightParams;
	bool useAlphaMask;
	bool editable;

	int maxOnScreen = 0;
	std::string trackingType;

	fs::path resourcesRoot;

	std::vector<std::shared_ptr<FX>> externFilters;
	std::vector<std::shared_ptr<GraphicsModel>> models;
	std::vector<FilterModule> filterModules;

	TCommonRenderParams commonRenderParams;

	std::array<std::string, 6> cubemapTextures;
	GLuint cubemapID = 0;

	std::map<int, bool> becameStable;

	std::map<size_t, std::vector<std::array<float, TARGET_DETAIL_LIMIT>>> confidencePastValuesMap;

	bool initialModulesRandomize = true;

	void updateModules(int moduleId, int targetIndex);

	std::array<size_t, MAX_TO_TRACK> initialModules = std::array<size_t, MAX_TO_TRACK> { 0, 0, 0, 0 };

	bool detectInside(std::vector<cv::Rect>& opencvFaces, cv::Rect rect);

	void resetStable();

	void outerTransform(TrackingTarget& target, ExternalRenderParams externalRenderParams);
	void outerDraw(TrackingTarget& target, ExternalRenderParams &externalRenderParams);
	void outerSetCMID(GLuint cmId);

	virtual void load();
	virtual void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams);
	virtual void draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams);
	virtual void unload();

	virtual void drawParticles(Matrix4f& projection);

	virtual void applyModule(FilterModule *module, size_t targetIndex, bool loadTexturesImmediately);

	virtual void onInputFrameResized();

	void loadFromJSON(const fs::path& path, FilterUiModel* externalInfo = nullptr);
	void saveToJSON(const fs::path& filterFolder, ExtraFilterData &extraData);

	virtual void transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *model);

	std::vector<std::shared_ptr<ParticleEffect>> particleEffects;
};
