#pragma once

#include <boost/variant.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <array>
#include <memory>

#include <GLSL/GLSL.h>
#include <FrameManager.h>
#include <ResourceManager.h>
#include <Particles/Particles.h>

#include <Common/CommonClasses.h>
#include <Common/JSONVectorReader.h>
#include <Common/Uniforms.h>
#include <Graphics/GraphicsCommon.h>
#include <Tracking/_Constants.h>
#include <Tracking/TrackingTarget.h>
#include <Common/Resolutions.h>

extern FrameManager frameManager;
extern ResourceManager resourceManager;
extern ShaderManagerWrapper shaderManagerWrapper;

class GraphicsModel;
class ObjectRenderParams;

typedef std::vector<std::shared_ptr<ObjectRenderParams>> TCommonRenderParams;

namespace pt = boost::property_tree;

class ModelRenderParams
{
public:
	ModelRenderParams()
	{

	}

	std::array<ValueSmoother, MAX_TO_TRACK> rollSmoother;
	std::array<ValueSmoother, MAX_TO_TRACK> pitchSmoother;
	std::array<ValueSmoother, MAX_TO_TRACK> yawSmoother;
	std::array<ValueSmoother, MAX_TO_TRACK> xCenterSmoother;
	std::array<ValueSmoother, MAX_TO_TRACK> yCenterSmoother;
	std::array<ValueSmoother, MAX_TO_TRACK> widthSmoother;

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
	std::shared_ptr<cwc::glShader> lightingShader;
	std::shared_ptr<cwc::glShader> blurShader;
	std::shared_ptr<cwc::glShader> bloomShader;

	std::pair <std::string, std::string> shadersSources;

	bool alphaTest;
	bool bloomEnabled;
	bool depthMask;
	bool depthTest;
	bool cullFace;
	bool blend;
	bool visible;

	std::string normalMap;
	GLuint normalMapID;
	std::string diffuseTexture;
	GLuint diffuseTextureID;
	std::string specularTexture;
	GLuint specularTextureID;
	std::string emissiveTexture;
	GLuint emissiveTextureID;

	Eigen::Vector4f cameraPos;
	Eigen::Vector4f lightPos;

	Eigen::Vector3f ambientLight;
	Eigen::Vector3f diffuseLight;
	Eigen::Vector3f bloomLight;
	float bloomPower;
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

class GraphicsModel;

class ShaderSetter {
public:

	ShaderSetter();

	ShaderSetter(std::shared_ptr<cwc::glShader> newShader, GraphicsModel& model);

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
	std::string name;
	ParticleEffect effect;
	Vector3f autoCoords;
	Vector3f coords;
	float scale;
	float autoScale;
	Vector3f globalGravity;
	Vector3f autoGlobalGravity;
	bool always_enabled;
	bool enable_rotation;
	bool enable_effect_gravity;
	bool enable_global_gravity;

	Vector3f lastCoords;

	JoinedParticleEffect();
	JoinedParticleEffect(const JoinedParticleEffect& other);
	void parse(pt::ptree& record);
};

class GraphicsModel
{
public:

	GraphicsModel();

	std::string name;

	GLuint cubeMapID;

	std::vector<GraphicsData> objects;
	std::array<std::vector<std::shared_ptr<ObjectRenderParams>>, MAX_TO_TRACK> objectRenderParams;
	ModelRenderParams renderParams;

	float depth = 0;

	std::array<bool, MAX_TO_TRACK> visible;
	std::array<bool, MAX_TO_TRACK> useAlphaMask;
	std::array<bool, MAX_TO_TRACK> animateTextures;
	std::array<size_t, MAX_TO_TRACK> animateTexturesCount;
	std::array<size_t, MAX_TO_TRACK> animateTexturesSpeed;

	bool canSwapSuit = true;

	bool useHardCodedUniforms = true;

	void SetUniformsForObject(cwc::glShader& shader, size_t i, size_t targetId);

	template<class T>
	static std::shared_ptr<GraphicsModel> create()
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
	virtual void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams) = 0;
	virtual void draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams) = 0;
	virtual void unload();
	virtual void postprocess(TrackingTarget& target);
	virtual void setCMID(GLuint cmId);

	virtual void transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *model);

	virtual std::shared_ptr<ObjectRenderParams> createDefaultObjectRenderParams();
	virtual void setDefaultObjectRenderParams(ObjectRenderParams& params);

	typedef std::pair<std::string, std::pair<std::string, std::string>> DefaultShadersMapElement;
	static const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> DefaultShadersMap;

	std::array<std::vector<JoinedParticleEffect>, MAX_TO_TRACK> particles;
};
