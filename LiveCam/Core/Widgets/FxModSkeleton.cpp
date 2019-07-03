#include <Widgets/FxModSkeleton.h>
#include <Common/JSONVectorReader.h>

#define _USE_MATH_DEFINES
#include <math.h>

const std::map<std::string, int> bonesNamesMap = 
{
	{ "spine", 0 },
	{ "neck", 1 },
	{ "head", 2 },
	{ "leftShoulder", 3 },
	{ "rightShoulder", 4 },
	{ "leftArm", 5 },
	{ "rightArm", 6 },
	{ "leftForeArm", 7 },
	{ "rightForeArm", 8 },
	{ "leftHip", 9 },
	{ "rightHip", 10 },
	{ "leftThigh", 11 },
	{ "rightThigh", 12 },
	{ "leftShin", 13 },
	{ "rightShin", 14 }
};

FaceBone getFaceBone(std::array<float, TARGET_DETAIL_LIMIT * 2> pts, int beginIndex, int endIndex)
{
	FaceBone result;

	result.begin = { pts[beginIndex * 2], pts[beginIndex * 2 + 1] };
	result.end = { pts[endIndex * 2], pts[endIndex * 2 + 1] };
	result.delta = result.end - result.begin;
	result.angle = atan2(result.delta[1], result.delta[0]);
	result.length = result.delta.norm();

	return result;
}

BoneTransformation Bone::calcTransformation(FaceBone& faceBone, bool bonesStretching)
{
	BoneTransformation result;

	Eigen::Vector2f relativeDirection = rotationBackward * faceBone.delta;

	float rotationAngle = atan2f(relativeDirection[1], relativeDirection[0]);

	float cosf_ = cosf(rotationAngle);
	float sinf_ = sinf(rotationAngle);

	result.rotation <<
		cosf_, -sinf_,
		sinf_, cosf_;

	result.rotation3x3 <<
		cosf_, -sinf_, 0,
		sinf_, cosf_, 0,
		0, 0, 1;

	result.shift = bonesStretching ? faceBone.delta * (1 - 1 / faceBone.length * length) : Vector2f(0, 0);

	result.joint = faceBone.begin;

	return result;
}

inline void calcBoneParams(Bone& bone)
{
	bone.direction = bone.end - bone.begin;
	bone.length = bone.direction.norm();
	bone.direction /= bone.length;

	float rotationAngle = atan2f(bone.direction[1], bone.direction[0]);

	float cosf_ = cosf(rotationAngle);
	float sinf_ = sinf(rotationAngle);

	bone.rotationBackward <<
		cosf_, sinf_,
		-sinf_, cosf_;
}

void Skeleton::initMissedBones()
{
	if (bones[1].index == -1)
	{
		auto& bone = bones[1];

		bone.name = "neck";
		bone.index = 1;

		bone.begin = bones[0].end;
		bone.end = bones[2].begin;

		calcBoneParams(bone);
	}

	if (bones[3].index == -1)
	{
		auto& bone = bones[3];

		bone.name = "leftShoulder";
		bone.index = 3;

		bone.begin = bones[0].end;
		bone.end = bones[5].begin;

		calcBoneParams(bone);
	}

	if (bones[4].index == -1)
	{
		auto& bone = bones[4];

		bone.name = "rightShoulder";
		bone.index = 4;

		bone.begin = bones[0].end;
		bone.end = bones[6].begin;

		calcBoneParams(bone);
	}

	if (bones[9].index == -1)
	{
		auto& bone = bones[9];

		bone.name = "leftHip";
		bone.index = 9;

		bone.begin = bones[0].begin;
		bone.end = bones[11].begin;

		calcBoneParams(bone);
	}

	if (bones[10].index == -1)
	{
		auto& bone = bones[10];

		bone.name = "rightHip";
		bone.index = 10;

		bone.begin = bones[0].begin;
		bone.end = bones[12].begin;

		calcBoneParams(bone);
	}
}

void Skeleton::load(pt::ptree& bonesJson, pt::ptree& weightsJson, const ObjectData& meshData)
{
	for (int i = 0; i < BONES_COUNT; ++i)
	{
		bones[i].index = -1;
	}

	for (auto &boneTree : bonesJson)
	{
		std::string boneName = boneTree.second.get<std::string>("name");

		int boneIndex = bonesNamesMap.at(boneName);

		auto& bone = bones[boneIndex];

		bone.index = boneIndex;
		bone.name = boneName;

		bone.begin = JSONVectorReader::readVector2f(boneTree.second.get_child("begin"));
		bone.end = JSONVectorReader::readVector2f(boneTree.second.get_child("end"));

		calcBoneParams(bone);
	}

	initMissedBones();

	noseTop = bones[0].begin;
	chinToNoseTopLength = (bones[0].begin - bones[0].end).norm();

	std::vector<Weights> indexedVertexes;

	for (auto &boneTree : weightsJson)
	{
		std::string boneName = boneTree.second.get<std::string>("name");

		int boneIndex = bonesNamesMap.at(boneName);

		auto& bone = bones[boneIndex];

		for (auto &weightRecord : boneTree.second.get_child("weights"))
		{
			int index = weightRecord.second.get<int>("index");

			if (indexedVertexes.size() < index + 1)
			{
				indexedVertexes.resize(index + 1);
			}

			auto& ref = indexedVertexes[index];

			ref.boneIndex[ref.weightsCount] = boneIndex;
			ref.weight[ref.weightsCount] = weightRecord.second.get<float>("weight");

			if (++ref.weightsCount == MAX_WEIGHTS_COUNT)
			{
				throw std::exception("Weights count outside of range!");
			}
		}
	}

	vertexes.resize(meshData.indexes.size());

	for (int i = 0; i < vertexes.size(); ++i)
	{
		vertexes[i] = indexedVertexes[meshData.indexes[i]];
	}

	glGenBuffers(1, &weightsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, weightsBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertexes.size() * sizeof(Weights), vertexes.data(), GL_STATIC_DRAW);
}

void Skeleton::unload()
{
	vertexes.clear();

	glDeleteBuffers(1, &weightsBuffer);
}

inline Eigen::Vector2f correctLength(FaceBone& faceBone, float length)
{
	float factor = length / faceBone.length;

	auto oldDelta = faceBone.delta;

	faceBone.delta = faceBone.delta * factor;
	faceBone.end = faceBone.begin + faceBone.delta;
	faceBone.length = length;

	return oldDelta * (factor - 1);
}

inline void correctPosition(FaceBone& faceBone, Vector2f& delta)
{
	faceBone.begin += delta;
	faceBone.end += delta;
}

void Skeleton::removeBonesStretching(std::array<FaceBone, BONES_COUNT>& faceBones)
{
}

std::array<BoneTransformation, BONES_COUNT> Skeleton::calcTransformations(TrackingTarget& face, bool bonesStretching)
{
	std::array<FaceBone, BONES_COUNT> faceBones =
	{
		getFaceBone(face.pts, 27, 30),
		getFaceBone(face.pts, 30, 51),
		getFaceBone(face.pts, 51, 57),
		getFaceBone(face.pts, 30, 31),
		getFaceBone(face.pts, 30, 35),
		getFaceBone(face.pts, 31, 48),
		getFaceBone(face.pts, 35, 54),
		getFaceBone(face.pts, 48, 7),
		getFaceBone(face.pts, 54, 9),
		getFaceBone(face.pts, 27, 21),
		getFaceBone(face.pts, 27, 22),
		getFaceBone(face.pts, 21, 19),
		getFaceBone(face.pts, 22, 24),
		getFaceBone(face.pts, 19, 17),
		getFaceBone(face.pts, 24, 26)
	};

	for (auto& faceBone : faceBones)
	{
		faceBone.begin[1] = -faceBone.begin[1];
		faceBone.end[1] = -faceBone.end[1];
		faceBone.delta[1] = -faceBone.delta[1];
	}
	float skeletonScale = (faceBones[0].begin - faceBones[0].end).norm() / chinToNoseTopLength;

	Eigen::Vector2f skeletonShift = faceBones[0].begin;

	for (auto& faceBone : faceBones)
	{
		faceBone.begin -= skeletonShift;
		faceBone.end -= skeletonShift;

		faceBone.begin /= skeletonScale;
		faceBone.end /= skeletonScale;
		faceBone.delta /= skeletonScale;
		faceBone.length /= skeletonScale;
	}

	if (!bonesStretching)
	{
		removeBonesStretching(faceBones);
	}

	std::array<BoneTransformation, BONES_COUNT> result;

	for (int i = 0; i < BONES_COUNT; ++i)
	{
		result[i] = bones[i].calcTransformation(faceBones[i], bonesStretching);
	}

	return result;
}
