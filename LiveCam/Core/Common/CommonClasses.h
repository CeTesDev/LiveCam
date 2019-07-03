#pragma once

#include <math.h>
#include <chrono>
#include <memory>
#include <array>
#include <vector>
#include <map>
#include <string>
#include <QDebug>
#include <QPixmap>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include <Eigen/Geometry>

#include <GLSL/GLSL.h>
#include <Tracking/_Constants.h>
#include "CVImageUtil.h"
#include "ValueSmoother.h"

namespace fs = boost::filesystem;

struct ObjectData
{
	std::vector<Eigen::Vector3f> vb;
	std::vector<Eigen::Vector3f> cb;
	std::vector<Eigen::Vector3f> nb;
	std::vector<Eigen::Vector2f> tb;

	std::vector<Eigen::Vector3f> tangentb;
	std::vector<Eigen::Vector3f> bitangentb;

	std::vector<int> indexes;
};

struct ExternalRenderParams
{
public:
    int frameWidth;
    int frameHeight;
    int trackedWidth;
    int trackedHeight;
    double vAngle;
    double zNear;
    double zFar;
    float timeShift;
    CVImageUtil::FrameLightParams frameParams;
    CVImageUtil::FaceLightParams faceParams;
};

class ImagePoints
{
public:
    ImagePoints(std::array<float, TARGET_DETAIL_LIMIT * 2>& inPts)
        : pts(inPts)
    {

    }
    cv::Point2f at(int i)
    {
        return cv::Point2f(pts[(i - 1) * 2], pts[(i - 1) * 2 + 1]);
    }
    Eigen::Vector2f getEigenAt(int i)
    {
        return Eigen::Vector2f(pts[(i - 1) * 2], pts[(i - 1) * 2 + 1]);
    }
    std::array<float, TARGET_DETAIL_LIMIT * 2>& pts;
};

class ShaderManagerWrapper
{
public:

    std::shared_ptr<cwc::glShader> LoadFromFile(const char *vertexFile, const char *fragmentFile);

    void Clear();

protected:

    std::map<std::string, std::shared_ptr<cwc::glShader>> cachedShaders;

    std::shared_ptr<cwc::glShaderManager> shaderManager;
};

struct ExtraFilterData;

struct ExtraRenderParamsData
{
    size_t shaderIndex;
    ExtraFilterData *root;

    ExtraRenderParamsData(ExtraFilterData *root)
    {
        this->root = root;
    }
};

struct ExtraModelData
{
    std::vector<ExtraRenderParamsData> renderParams;
    ExtraFilterData *root;

    ExtraModelData(ExtraFilterData *root)
    {
        this->root = root;
    }
};

struct ExtraModuleData
{
    QPixmap moduleIcon;
    std::string moduleIconPath;

    std::vector<ExtraModelData> models;

    ExtraFilterData *root;

    ExtraModuleData(ExtraFilterData *root)
    {
        this->root = root;
    }
};

class FX;

struct ExtraFilterData
{
    fs::path resourcesRoot;
    fs::path filterFolder;

    std::vector<ExtraModuleData> modules;
    std::vector<ExtraRenderParamsData> renderParams;

    QPixmap filterIcon;
    std::string filterIconPath;

    std::string title;

    std::shared_ptr<FX> filter;
};
