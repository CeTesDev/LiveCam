#pragma once

#include <boost/variant.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <array>
#include <memory>

#include "../glsl.h"
#include <FrameManager.h>
extern FrameManager frameManager;

class VulkanModel;
class ObjectRenderParams;

typedef std::vector<std::shared_ptr<ObjectRenderParams>> TCommonRenderParams;

#include <Common/JSONVectorReader.h>
#include <Common/Uniforms.h>
#include <FXModel.h>
#include <Common/Resolutions.h>
#include <Common/CommonClasses.h>

extern boost::format StructUniformFieldFormatter1;
extern boost::format StructUniformFieldFormatter2;

#define StructUniformField1(name, index, field) \
	const_cast<char*>((StructUniformFieldFormatter1 % name % index % field).str().c_str())

#define StructUniformField2(name, index1, field1, index2, field2) \
	const_cast<char*>((StructUniformFieldFormatter2 % name % index1 % field1 % index2 % field2).str().c_str()

class ModelRenderParams
{
public:
	ModelRenderParams()
	{

	}

	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> rollSmoother;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> pitchSmoother;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> yawSmoother;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> xCenterSmoother;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> yCenterSmoother;
	std::array<ValueSmoother, ObjectTracker::MAX_TO_TRACK> widthSmoother;

	Eigen::Matrix4f modelView;
	Eigen::Matrix4f projection;
	Eigen::Matrix3f yawMatrix;
	Eigen::Matrix3f rotationMatrix;

	std::array<Eigen::Matrix4f, 1> additionalMatrices4;

	void SetUniforms(cwc::glShader& shader);
};

class ObjectRenderParams
{
public:
	std::shared_ptr<cwc::glShader> shader;
	std::pair <std::string, std::string> shadersSources;

	bool alphaTest;
	bool depthMask;
	bool depthTest;
	bool cullFace;
	bool blend;
	bool visible;

	std::string normalMap;
	GLuint normalMapID;

	Eigen::Vector4f cameraPos;
	Eigen::Vector4f lightPos;

	Eigen::Vector3f ambientLight;
	Eigen::Vector3f diffuseLight;
	Eigen::Vector3f specularLight;
	float specularPower;

	Eigen::Vector4f materialColor;

	float reflectionRatio;

	std::map<std::string, TUniformVariant> additionalUniforms;

	ObjectRenderParams();

	ObjectRenderParams(boost::property_tree::ptree& renderParams, fs::path& resourcesRoot);

	void reset();
	void applyMTL(boost::property_tree::ptree &MTLtree);
	boost::property_tree::ptree getPTree(ExtraRenderParamsData &data);

	bool operator == (ObjectRenderParams &other) const;
};

class VulkanModel;

class ShaderSetter {
public:

	ShaderSetter();

	ShaderSetter(std::shared_ptr<cwc::glShader> newShader, VulkanModel& model);

	ShaderSetter(const ShaderSetter& from) = delete;

	ShaderSetter(ShaderSetter&& from);

	ShaderSetter& operator=(const ShaderSetter& from) = delete;

	ShaderSetter& operator=(ShaderSetter&& from);

	~ShaderSetter();

protected:
	std::shared_ptr<cwc::glShader> shader;

};

class VertexAttribSetter
{
public:
	VertexAttribSetter();

	VertexAttribSetter(const VertexAttribSetter& from) = delete;

	VertexAttribSetter(VertexAttribSetter&& from);

	VertexAttribSetter& operator=(const VertexAttribSetter& from) = delete;

	VertexAttribSetter& operator=(VertexAttribSetter&& from);

	VertexAttribSetter(cwc::glShader& shader, char* attribName, GLuint attribType, int attribSize, GLuint buffer, int offset = 0, int stride = 0);

	~VertexAttribSetter();

	static VertexAttribSetter PositionAttribSetter(cwc::glShader& shader, GLuint buffer);
	static VertexAttribSetter TexCoordAttribSetter(cwc::glShader& shader, GLuint buffer);
	static VertexAttribSetter ColorAttribSetter(cwc::glShader& shader, GLuint buffer);
	static VertexAttribSetter NormalAttribSetter(cwc::glShader& shader, GLuint buffer);
	static VertexAttribSetter TangentAttribSetter(cwc::glShader& shader, GLuint buffer);
	static VertexAttribSetter BitangentAttribSetter(cwc::glShader& shader, GLuint buffer);

protected:
	int loc = -1;
};

struct JoinedParticleEffect
{
	ParticleEffect effect;
	Vector3f autoCoords;
	Vector3f coords;
	float scale;
	float autoScale;
	Vector3f globalGravity;
	Vector3f autoGlobalGravity;
	bool enable_rotation;
	bool enable_effect_gravity;
	bool enable_global_gravity;

	Vector3f lastCoords;

	JoinedParticleEffect();
	JoinedParticleEffect(const JoinedParticleEffect& other);
	void parse(pt::ptree& record);
};

class VulkanModel
{
public:

	VulkanModel();

	std::string name;

	std::vector<VulkanDataObject> objects;
	std::array<std::vector<std::shared_ptr<ObjectRenderParams>>, ObjectTracker::MAX_TO_TRACK> objectRenderParams;
	ModelRenderParams renderParams;

	float depth = 0;

	std::array<bool, ObjectTracker::MAX_TO_TRACK> visible;
	std::array<bool, ObjectTracker::MAX_TO_TRACK> useAlphaMask;
	std::array<bool, ObjectTracker::MAX_TO_TRACK> animateTextures;
	std::array<size_t, ObjectTracker::MAX_TO_TRACK> animateTexturesCount;
	std::array<size_t, ObjectTracker::MAX_TO_TRACK> animateTexturesSpeed;

	bool canSwapSuit = true;

	bool useHardCodedUniforms = true;

	void SetUniformsForObject(cwc::glShader& shader, size_t i, size_t face);

	template<class T>
	static std::shared_ptr<VulkanModel> create()
	{
		return std::make_shared<T>();
	}

	static const std::string TYPE_NAME;
	virtual std::string getTypeName();

	virtual void onInputFrameResized();

	virtual void loadFromJSON(boost::property_tree::ptree& modelRecord);

	virtual boost::property_tree::ptree getPTree(ExtraModelData &data);
	virtual void prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData);

	boost::property_tree::ptree findMatchingRenderParamsIndexes(TCommonRenderParams &commonRenderParams, ExtraModelData& modelData);

	virtual void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
		bool loadTexturesImmediately);

	virtual bool load();
	virtual void transform(FXModel& face, ExternalRenderParams &externalRenderParams) = 0;
	virtual void draw(FXModel& face, ExternalRenderParams &externalRenderParams) = 0;
	virtual void unload();

	virtual void drawOnCVFrame_OLD(cv::Mat frame, FXModel& fxModel);

	virtual std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams();
	virtual void setDefaultObjectRenderParams(ObjectRenderParams& params);

	typedef std::pair<std::string, std::pair<std::string, std::string>> DefaultShadersMapElement;
	static const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> DefaultShadersMap;

	std::array<std::vector<JoinedParticleEffect>, ObjectTracker::MAX_TO_TRACK> particles;
};

#include "FX.h"
