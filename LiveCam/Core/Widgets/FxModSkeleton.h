#pragma once

#include <string>
#include <array>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

#include "Eigen/Geometry"
#include <Common/CommonClasses.h>
#include <Tracking/TrackingTarget.h>

using namespace std;
using namespace Eigen;

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

const int BONES_COUNT = 100;

extern const std::map<std::string, int> bonesNamesMap;

struct FaceBone
{
	Eigen::Vector2f begin;
	Eigen::Vector2f end;
	Eigen::Vector2f delta;
	double angle;
	double length;
};

struct BoneTransformation
{
	Eigen::Vector2f joint;
	Eigen::Matrix2f rotation;
	Eigen::Matrix3f rotation3x3;
	Eigen::Vector2f shift;
};

struct Bone
{
	int index;

	std::string name;

	float length;
	Eigen::Vector2f begin;
	Eigen::Vector2f end;
	Eigen::Vector2f direction;
	Eigen::Matrix2f rotationBackward;

	BoneTransformation calcTransformation(FaceBone& transformed, bool bonesStretching);
};

const int MAX_WEIGHTS_COUNT = 6;

struct Weights
{
	int weightsCount = 0;
	int boneIndex[MAX_WEIGHTS_COUNT];
	float weight[MAX_WEIGHTS_COUNT];
};

struct Skeleton
{
	GLuint weightsBuffer;

	std::vector<Weights> vertexes;
	std::array<Bone, BONES_COUNT> bones;
	Eigen::Vector2f noseTop;
	float chinToNoseTopLength;

	void initMissedBones();

	void load(pt::ptree& bonesJson, pt::ptree& weightsJson, const ObjectData& meshData);
	std::array<BoneTransformation, BONES_COUNT> calcTransformations(TrackingTarget& face, bool bonesStretching);

	void removeBonesStretching(std::array<FaceBone, BONES_COUNT>& faceBones);
	void unload();
};