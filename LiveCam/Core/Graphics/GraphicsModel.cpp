#include "GraphicsModel.h"

#include <unordered_map>
#include <chrono>

const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> GraphicsModel::DefaultShadersMap =
{
	{ "regular", { "default.vertex", "default.frag" } },
	{ "no texture", { "notexture.vertex", "notexture.frag" } },
	{ "normal map", { "normalmap.vertex", "normalmap.frag" } }
};

ObjectRenderParams::ObjectRenderParams()
	: cameraPos(0, 0, 0, 1)
	, lightPos(0, 0, 1, 0)
	, ambientLight(0.5, 0.5, 0.5)
	, diffuseLight(0.5, 0.5, 0.5)
	, specularLight(0, 0, 0)
	, bloomLight(0, 0, 0)
	, bloomPower(0)
	, specularPower(1)
	, materialColor(1, 1, 1, 1)
	, reflectionRatio(0)
	, shader(nullptr)
	, normalMapID(0)
	, diffuseTextureID(0)
	, specularTextureID(0)
	, emissiveTextureID(0)
	, depthTest(true)
	, depthMask(true)
	, alphaTest(false)
	, bloomEnabled(false)
	, cullFace(true)
	, blend(false)
	, visible(true)
{

}

ObjectRenderParams::ObjectRenderParams(boost::property_tree::ptree& renderParams, fs::path& resourcesRoot)
{
	shadersSources = { renderParams.get<std::string>("vertexShader"), renderParams.get<std::string>("fragmentShader") };
	shadersSources.first = shadersSources.first;
	shadersSources.second = shadersSources.second;
	shader = shaderManagerWrapper.LoadFromFile((resourcesRoot / shadersSources.first).string().data(),
		(resourcesRoot / shadersSources.second).string().data());

	lightingShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/lighting.vertex", "./Assets/shaders/fragment/lighting.frag");
	lightingShader->begin();
	lightingShader->setUniform1i("Texture", 0);
	lightingShader->setUniform1i("SpecularTexture", 5);
	lightingShader->setUniform1i("EmissiveTexture", 6);
	lightingShader->end();

	blurShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/blur.vertex", "./Assets/shaders/fragment/blur.frag");
	blurShader->begin();
	blurShader->setUniform1i("Texture", 0);
	blurShader->end();

	bloomShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/bloom.vertex", "./Assets/shaders/fragment/bloom.frag");
	bloomShader->begin();
	bloomShader->setUniform1i("Texture", 0);
	bloomShader->setUniform1i("GlowTexture", 1);
	bloomShader->end();

	visible = renderParams.get<bool>("visible", true);
	depthTest = renderParams.get<bool>("depthTest", true);
	depthMask = renderParams.get<bool>("depthMask", true);
	alphaTest = renderParams.get<bool>("alphaTest", false);
	bloomEnabled = renderParams.get<bool>("bloomEnabled", false);
	cullFace = renderParams.get<bool>("cullFace", true);
	blend = renderParams.get<bool>("blend", false);

	auto MTL = renderParams.get_child_optional("MTL");
	if (MTL)
	{
		applyMTL(MTL.get());
	}

	normalMap = renderParams.get<std::string>("normalMap", "");
	normalMapID = 0;
	diffuseTexture = renderParams.get<std::string>("diffuseTexture", "");
	diffuseTextureID = 0;
	specularTexture = renderParams.get<std::string>("specularTexture", "");
	specularTextureID = 0;
	emissiveTexture = renderParams.get<std::string>("emissiveTexture", "");
	emissiveTextureID = 0;


	additionalUniforms.clear();

	auto uniforms = renderParams.get_child_optional("uniforms");
	if (uniforms)
	{
		for (auto &uniform : uniforms.get())
		{
			auto record = JSONVectorReader::readUniform(uniform.second);
			additionalUniforms[record.first] = record.second;
		}
	}
}

void ObjectRenderParams::reset()
{
	cameraPos << 0, 0, 0, 1;
	lightPos << 0, 0, 1, 0;
	ambientLight << 0.5, 0.5, 0.5;
	diffuseLight << 0.5, 0.5, 0.5;
	bloomLight << 0, 0, 0;
	bloomPower = 0;
	specularLight << 0, 0, 0;
	specularPower = 1;
	materialColor << 1, 1, 1, 1;
	reflectionRatio = 0;
	normalMapID = 0;
	diffuseTextureID = 0;
	specularTextureID = 0;
	emissiveTextureID = 0;
	depthTest = true;
	depthMask = true;
	alphaTest = false;
	bloomEnabled = false;
	cullFace = true;
	blend = false;
	visible = true;
}

void ObjectRenderParams::applyMTL(boost::property_tree::ptree &MTLtree)
{
	auto tree = MTLtree.get_child_optional("lightPos");
	if (tree)
	{
		additionalUniforms["lightPos"] = JSONVectorReader::readVector4f(tree.get());
	}

	tree = MTLtree.get_child_optional("ambientLight");
	if (tree)
	{
		additionalUniforms["ambientLight"] = JSONVectorReader::readVector3f(tree.get());
	}

	tree = MTLtree.get_child_optional("diffuseLight");
	if (tree)
	{
		additionalUniforms["diffuseLight"] = JSONVectorReader::readVector3f(tree.get());
	}

	tree = MTLtree.get_child_optional("bloomLight");
	if (tree)
	{
		additionalUniforms["bloomLight"] = JSONVectorReader::readVector3f(tree.get());
	}

	auto bpRecord = MTLtree.get_optional<float>("bloomPower");
	if (bpRecord)
	{
		additionalUniforms["bloomPower"] = bpRecord.get();
	}

	tree = MTLtree.get_child_optional("specularLight");
	if (tree)
	{
		additionalUniforms["specularLight"] = JSONVectorReader::readVector3f(tree.get());
	}

	auto spRecord = MTLtree.get_optional<float>("specularPower");
	if (spRecord)
	{
		additionalUniforms["specularPower"] = spRecord.get();
	}

	auto nmRecord = MTLtree.get_optional<std::string>("normalMap");
	if (nmRecord)
	{
		normalMap = nmRecord.get();
	}

	auto atRecord = MTLtree.get_optional<std::string>("diffuseTexture");
	if (atRecord)
	{
		diffuseTexture = atRecord.get();
	}

	auto stRecord = MTLtree.get_optional<std::string>("specularTexture");
	if (stRecord)
	{
		specularTexture = stRecord.get();
	}

	auto etRecord = MTLtree.get_optional<std::string>("emissiveTexture");
	if (etRecord)
	{
		emissiveTexture = etRecord.get();
	}
}

boost::property_tree::ptree ObjectRenderParams::getPTree(ExtraRenderParamsData &data)
{
	boost::property_tree::ptree result;

	auto originFile = fs::path(data.root->resourcesRoot / shadersSources.first);
	auto copiedFile = fs::path(data.root->filterFolder / fs::path(shadersSources.first).filename());
	if (!equivalent(originFile, copiedFile))
	{
		fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
	}

	originFile = fs::path(data.root->resourcesRoot / shadersSources.second);
	copiedFile = fs::path(data.root->filterFolder / fs::path(shadersSources.second).filename());
	if (!equivalent(originFile, copiedFile))
	{
		fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
	}

	if (!normalMap.empty())
	{
		originFile = fs::path(data.root->resourcesRoot / normalMap);
		copiedFile = fs::path(data.root->filterFolder / fs::path(normalMap).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	if (!diffuseTexture.empty())
	{
		originFile = fs::path(data.root->resourcesRoot / diffuseTexture);
		copiedFile = fs::path(data.root->filterFolder / fs::path(diffuseTexture).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	if (!specularTexture.empty())
	{
		originFile = fs::path(data.root->resourcesRoot / specularTexture);
		copiedFile = fs::path(data.root->filterFolder / fs::path(specularTexture).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	if (!emissiveTexture.empty())
	{
		originFile = fs::path(data.root->resourcesRoot / emissiveTexture);
		copiedFile = fs::path(data.root->filterFolder / fs::path(emissiveTexture).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}

	shadersSources.first = fs::path(shadersSources.first).filename().string();
	shadersSources.second = fs::path(shadersSources.second).filename().string();
	normalMap = fs::path(normalMap).filename().string();
	diffuseTexture = fs::path(diffuseTexture).filename().string();
	specularTexture = fs::path(specularTexture).filename().string();
	emissiveTexture = fs::path(emissiveTexture).filename().string();

	result.put("vertexShader", shadersSources.first);
	result.put("fragmentShader", shadersSources.second);

	if (!visible) result.put("visible", "false");
	if (!depthTest) result.put("depthTest", "false");
	if (!cullFace) result.put("cullFace", "false");
	if (blend) result.put("blend", "true");
	if (alphaTest) result.put("alphaTest", "true");
	if (bloomEnabled) result.put("bloomEnabled", "true");

	if (!normalMap.empty()) result.put("normalMap", normalMap);
	if (!diffuseTexture.empty()) result.put("diffuseTexture", diffuseTexture);
	if (!specularTexture.empty()) result.put("specularTexture", specularTexture);
	if (!emissiveTexture.empty()) result.put("emissiveTexture", emissiveTexture);

	boost::property_tree::ptree uniformsTree;
	static ObjectRenderParamVisitor_toPTree visitor;

	for (auto uniform : additionalUniforms)
	{
		visitor.uniformName = uniform.first;
		uniformsTree.push_back(std::make_pair("", boost::apply_visitor(visitor, uniform.second)));
	}

	result.put_child("uniforms", uniformsTree);

	return result;
}

bool ObjectRenderParams::operator == (ObjectRenderParams &other) const
{
	return shader == other.shader &&
		depthMask == other.depthMask &&
		alphaTest == other.alphaTest &&
		bloomEnabled == other.bloomEnabled &&
		depthTest == other.depthTest &&
		blend == other.blend &&
		cullFace == other.cullFace &&
		visible == other.visible &&
		normalMap == other.normalMap &&
		diffuseTexture == other.diffuseTexture &&
		specularTexture == other.specularTexture &&
		emissiveTexture == other.emissiveTexture &&
		cameraPos == other.cameraPos &&
		lightPos == other.lightPos &&
		ambientLight == other.ambientLight	&&
		diffuseLight == other.diffuseLight &&
		bloomLight == other.bloomLight &&
		bloomPower == other.bloomPower &&
		specularLight == other.specularLight &&
		specularPower == other.specularPower && 
		materialColor == other.materialColor &&
		reflectionRatio == other.reflectionRatio &&
		additionalUniforms == other.additionalUniforms;
}

ShaderSetter::ShaderSetter()
{
}

ShaderSetter::ShaderSetter(ShaderSetter&& from)
{
	shader = from.shader;

	from.shader.reset();
}

ShaderSetter& ShaderSetter::operator=(ShaderSetter&& from)
{
	if (this != &from)
	{
		shader = from.shader;

		from.shader.reset();
	}
	return *this;
}

ShaderSetter::ShaderSetter(std::shared_ptr<cwc::glShader> newShader, GraphicsModel& model)
{
	shader = newShader;

	if (shader)
	{
		shader->begin();

		model.renderParams.SetUniforms(*shader);
	}

}

ShaderSetter::~ShaderSetter()
{
	if (shader)
	{
		shader->end();
	}
}

VertexAttribSetter::VertexAttribSetter()
{

}

VertexAttribSetter::VertexAttribSetter(VertexAttribSetter&& from)
	: loc(from.loc)
{
	from.loc = -1;
}

VertexAttribSetter& VertexAttribSetter::operator=(VertexAttribSetter&& from)
{
	loc = from.loc;

	from.loc = -1;

	return *this;
}

VertexAttribSetter::VertexAttribSetter(cwc::glShader& shader, char* attribName, GLuint attribType, int attribSize, GLuint buffer, int offset, int stride)
{
	loc = shader.getVetrexShaderAttribLocation(attribName);
	if (loc >= 0)
	{
		glEnableVertexAttribArray(loc);
		shader.BindAttribLocation(loc, attribName);

		glBindBuffer(GL_ARRAY_BUFFER, buffer);

		switch (attribType)
		{
		case GL_FLOAT:
			glVertexAttribPointer(loc, attribSize, attribType, GL_FALSE, stride, (GLvoid*)(size_t)offset);
			break;

		case GL_INT:
			glVertexAttribIPointer(loc, attribSize, attribType, stride, (GLvoid*)(size_t)offset);
			break;
		}
	}
}

VertexAttribSetter::~VertexAttribSetter()
{
	if (loc >= 0)
	{
		glDisableVertexAttribArray(loc);
	}
}

VertexAttribSetter VertexAttribSetter::PositionAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return VertexAttribSetter(shader, "VertexPosition", GL_FLOAT, 3, buffer);
}

VertexAttribSetter VertexAttribSetter::TexCoordAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return (VertexAttribSetter(shader, "VertexTexCoord", GL_FLOAT, 2, buffer));
}

VertexAttribSetter VertexAttribSetter::ColorAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return (VertexAttribSetter(shader, "VertexColor", GL_FLOAT, 4, buffer));
}

VertexAttribSetter VertexAttribSetter::NormalAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return (VertexAttribSetter(shader, "VertexNormal", GL_FLOAT, 3, buffer));
}

VertexAttribSetter VertexAttribSetter::TangentAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return (VertexAttribSetter(shader, "VertexTangent", GL_FLOAT, 3, buffer));
}

VertexAttribSetter VertexAttribSetter::BitangentAttribSetter(cwc::glShader& shader, GLuint buffer)
{
	return (VertexAttribSetter(shader, "VertexBitangent", GL_FLOAT, 3, buffer));
}

void ModelRenderParams::SetUniforms(cwc::glShader& shader)
{
	shader.setUniformMatrix4fv("ModelViewMatrix", 1, false, modelView.data());
	shader.setUniformMatrix4fv("ProjectionMatrix", 1, false, projection.data());
	//shader.setUniformMatrix3fv("rotate3x3", 1, false, yawMatrix.data());
	//shader.setUniformMatrix3fv("ModelViewMatrix3x3", 1, false, modelView.block<3, 3>(0, 0).data());
	//shader.setUniformMatrix3fv("RotationMatrix", 1, false, rotationMatrix.data());
	//shader.setUniformMatrix4fv("CompressingModelViewMatrix", 1, false, additionalMatrices4[0].data());



	GLint uniformLocation = -1;

	uniformLocation = shader.GetUniformLocation("Texture");

	if (uniformLocation != -1)
	{
		glUniform1i(uniformLocation, 0);
	}

	/*uniformLocation = shader.GetUniformLocation("SpecularTexture");

	if (uniformLocation != -1)
	{
		glUniform1i(uniformLocation, 2);
	}

	uniformLocation = shader.GetUniformLocation("NormalMap");

	if (uniformLocation != -1)
	{
		glUniform1i(uniformLocation, 2);
	}*/
}

const std::string GraphicsModel::TYPE_NAME = "GraphicsModel";

std::string GraphicsModel::getTypeName()
{
	return TYPE_NAME;
}

JoinedParticleEffect::JoinedParticleEffect() {}
JoinedParticleEffect::JoinedParticleEffect(const JoinedParticleEffect& other) {}

void JoinedParticleEffect::parse(pt::ptree& record)
{
	std::string path;
	try
	{
		path = record.get<std::string>("JSONsource");

		pt::ptree effectTree;
		boost::property_tree::json_parser::read_json(path, effectTree);

		effect.parse(effectTree);
	}
	catch (...)
	{
		qDebug().noquote() << "\nFAILED TO LOAD PARTICLE EFFECT: " << path.c_str() << " \n";
		return;
	}

	name = record.get<std::string>("name", "");

	auto optional = record.get_child_optional("coords");
	effect.setCoords(optional ? JSONVectorReader::readVector3f(optional.get()) : Vector3f(0, 0, 0));
	coords << effect.getCoords();
	autoCoords << coords;

	always_enabled = record.get<bool>("always_enabled", false);
	enable_rotation = record.get<bool>("enable_rotation", false);
	enable_effect_gravity = record.get<bool>("enable_effect_gravity", true);
	enable_global_gravity = record.get<bool>("enable_global_gravity", true);

	effect.setDepthTest(record.get<bool>("depthTest", true));

	optional = record.get_child_optional("globalGravity");
	globalGravity = optional ? JSONVectorReader::readVector3f(optional.get()) : Vector3f(0, 0, 0);
	autoGlobalGravity = globalGravity;
		
	scale = record.get<float>("scale", 1);
}

GraphicsModel::GraphicsModel()
{
	visible.fill(true);
	useAlphaMask.fill(false);
	animateTextures.fill(false);
	animateTexturesSpeed.fill(0);
	animateTexturesCount.fill(0);
	useHardCodedUniforms = true;
}

void GraphicsModel::setCMID(GLuint cmId)
{
	cubeMapID = cmId;
}

void GraphicsModel::onInputFrameResized()
{
	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		renderParams.xCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		renderParams.yCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		renderParams.widthSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		renderParams.pitchSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		renderParams.yawSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		renderParams.rollSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
	}
}

void GraphicsModel::SetUniformsForObject(cwc::glShader& shader, size_t i, size_t face)
{
	if (objectRenderParams[face].size() == 0)
	{
		return;
	}

	size_t index = i;

	if (index > objectRenderParams[face].size() - 1)
	{
		index = objectRenderParams[face].size() - 1;
	}

	glDepthMask(objectRenderParams[face][index]->depthMask);

	if (objectRenderParams[face][index]->alphaTest)
	{
		glEnable(GL_ALPHA_TEST);
	}
	else
	{
		glDisable(GL_ALPHA_TEST);
	}

	if (objectRenderParams[face][index]->depthTest)
	{
		glEnable(GL_DEPTH_TEST);
	}
	else
	{
		glDisable(GL_DEPTH_TEST);
	}

	if (objectRenderParams[face][index]->blend)
	{
		glEnable(GL_BLEND);
	}
	else
	{
		glDisable(GL_BLEND);
	}

	if (objectRenderParams[face][index]->cullFace)
	{
		glEnable(GL_CULL_FACE);
	}
	else
	{
		glDisable(GL_CULL_FACE);
	}
	
	if (useHardCodedUniforms)
	{
		shader.setUniform1i("Texture", objectRenderParams[face][index]->diffuseTextureID);
		shader.setUniform1i("SpecularTexture", objectRenderParams[face][index]->specularTextureID);
		shader.setUniform1i("CubeMap", cubeMapID);
		shader.setUniform4fv("cameraPos", 1, &objectRenderParams[face][index]->cameraPos[0]);
		shader.setUniform4fv("lightPos", 1, &objectRenderParams[face][index]->lightPos[0]);
		shader.setUniform3fv("ambientLight", 1, &objectRenderParams[face][index]->ambientLight[0]);
		shader.setUniform3fv("diffuseLight", 1, &objectRenderParams[face][index]->diffuseLight[0]);
		//shader.setUniform3fv("bloomLight", 1, &objectRenderParams[face][index]->bloomLight[0]);
		//shader.setUniform1f("bloomPower", objectRenderParams[face][index]->bloomPower);
		shader.setUniform3fv("specularLight", 1, &objectRenderParams[face][index]->specularLight[0]);
		shader.setUniform1f("specularPower", objectRenderParams[face][index]->specularPower);
		//shader.setUniform1f("reflectionRatio", objectRenderParams[face][index]->reflectionRatio);
		//shader.setUniform4fv("materialColor", 1, &objectRenderParams[face][index]->materialColor[0]);
	}

	static ObjectRenderParamVisitor_GLSetter visitor;

	visitor.usedShader = objectRenderParams[face][index]->shader;

	for (auto &uniform : objectRenderParams[face][index]->additionalUniforms)
	{
		visitor.uniformName = uniform.first;
		boost::apply_visitor(visitor, uniform.second);
	}
}

void GraphicsModel::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	name = modelRecord.get<std::string>("name", "");
	canSwapSuit = modelRecord.get<bool>("canSwapSuit", true);

	auto particleList = modelRecord.get_child_optional("particles");
	if (particleList)
	{
		for (int i = 0; i < MAX_TO_TRACK; ++i)
		{
			particles[i].resize(particleList.get().size());
			int p = 0;
			for (auto &particle : particleList.get())
			{
				particles[i][p++].parse(particle.second);
			}
		}
	}
}

boost::property_tree::ptree GraphicsModel::getPTree(ExtraModelData &data)
{
	boost::property_tree::ptree result;

	result.put("name", name);

	if (!canSwapSuit)
	{
		result.put("canSwapSuit", "false");
	}

	return result;
}

boost::property_tree::ptree GraphicsModel::findMatchingRenderParamsIndexes(TCommonRenderParams &commonRenderParams,
	ExtraModelData& modelData)
{
	boost::property_tree::ptree renderIDs;

	auto renderParams = modelData.renderParams.begin();

	for (auto target : objectRenderParams[0])
	{
		auto ptr = std::find_if(commonRenderParams.begin(), commonRenderParams.end(), [target](std::shared_ptr<ObjectRenderParams> A)
			-> bool {
			return *target == *A;
		});

		int renderParamsID;

		if (ptr == commonRenderParams.end())
		{
			renderParamsID = commonRenderParams.size();
			commonRenderParams.push_back(target);
			modelData.root->renderParams.push_back(*renderParams++);
		}
		else
		{
			renderParamsID = std::distance(commonRenderParams.begin(), ptr);
		}

		boost::property_tree::ptree renderID;
		renderID.put("", renderParamsID);

		renderIDs.push_back(std::make_pair("", renderID));
	}

	return renderIDs;
}

void GraphicsModel::prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData)
{
	
}

void GraphicsModel::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	visible[targetIndex] = suit.get<bool>("visible", true);
	useAlphaMask[targetIndex] = suit.get<bool>("useAlphaMask", false);
	animateTextures[targetIndex] = suit.get<bool>("animateTextures", false);
}

bool GraphicsModel::load()
{
	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		for (auto &particle : particles[i])
		{
			particle.lastCoords = { 0, 0, 0 };
			particle.effect.setDirection({ 0, 0 });
			particle.effect.setSpeed({ 0, 0, 0 });

			particle.effect.load();
			particle.effect.startSpawn();
		}
	}

	return true;
}

void GraphicsModel::unload()
{
	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		for (auto &particle : particles[i])
		{
			particle.effect.unload();
			particle.effect.freeze();
		}
	}

	for (auto& obj : objects)
	{
		if (obj.vb != 0)
		{
		 	glDeleteBuffers(1, &obj.vb);
		}
		 
		if (obj.nb != 0)
		{
		 	glDeleteBuffers(1, &obj.nb);
		}
		 
		if (obj.cb != 0)
		{
		 	glDeleteBuffers(1, &obj.cb);
		}
		 
		if (obj.tb != 0)
		{
		 	glDeleteBuffers(1, &obj.tb);
		}
		 
		if (obj.tangentb != 0)
		{
		 	glDeleteBuffers(1, &obj.tangentb);
		}
		 
		if (obj.bitangentb != 0)
		{
		 	glDeleteBuffers(1, &obj.bitangentb);
		}
	}

	objects.clear();
}

void GraphicsModel::postprocess(TrackingTarget& target)
{
	if ((TYPE_NAME != "3DModel" && TYPE_NAME != "3DAnimatedModel") || objects.size() < 1)
	{
		return;
	}

	//glPushAttrib(GL_ALL_ATTRIB_BITS);

	//auto prevColorBuffer = frameManager.getCurrentDrawBuffer();
	//frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::GLOW_MASK);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	//glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (size_t i = 0; i < objects.size(); i++)
	{
		GraphicsData o = objects[i];
		if (o.vb < 1)
		{
			continue;
		}

		int index = i < objectRenderParams[target.pointId].size() ? i : objectRenderParams[target.pointId].size() - 1;
		ObjectRenderParams *params = objectRenderParams[target.pointId][index].get();
		if (params->specularTextureID == 0 && params->emissiveTextureID == 0)
		{
			continue;
		}

		params->lightingShader->begin();
		//renderParams.SetUniforms(*params->lightingShader);
		//SetUniformsForObject(*params->lightingShader, i, target.pointId);
		params->lightingShader->setUniformMatrix4fv("ProjectionMatrix", 1, false, renderParams.projection.data());
		params->lightingShader->setUniformMatrix4fv("ModelViewMatrix", 1, false, renderParams.modelView.data());
		params->lightingShader->setUniform1i("Texture", params->diffuseTextureID);
		params->lightingShader->setUniform1i("SpecularTexture", params->specularTextureID);
		params->lightingShader->setUniform1i("EmissiveTexture", params->emissiveTextureID);
		params->lightingShader->setUniform4fv("cameraPos", 1, &params->cameraPos[0]);
		params->lightingShader->setUniform4fv("lightPos", 1, &params->lightPos[0]);
		params->lightingShader->setUniform3fv("ambientLight", 1, &params->ambientLight[0]);
		params->lightingShader->setUniform3fv("diffuseLight", 1, &params->diffuseLight[0]);
		//shader.setUniform3fv("bloomLight", 1, &params->bloomLight[0]);
		//shader.setUniform1f("bloomPower", params->bloomPower);
		params->lightingShader->setUniform3fv("specularLight", 1, &params->specularLight[0]);
		params->lightingShader->setUniform1f("specularPower", params->specularPower);
		//shader.setUniform1f("reflectionRatio", objectRenderParams[face][index]->reflectionRatio);
		//params->lightingShader->setUniform4fv("materialColor", 1, &params->materialColor[0]);
		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*params->lightingShader, o.vb));
		VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*params->lightingShader, o.tb));

		//glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, params->diffuseTextureID);
		if (params->specularTextureID != 0)
		{
			glActiveTexture(GL_TEXTURE5);
			glBindTexture(GL_TEXTURE_2D, params->specularTextureID);
		}
		if (params->emissiveTextureID != 0)
		{
			glActiveTexture(GL_TEXTURE6);
			glBindTexture(GL_TEXTURE_2D, params->emissiveTextureID);
		}
		glActiveTexture(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapID);

		glBindVertexArray(o.va);
		glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
		glBindVertexArray(0);

		glFinish();

		//params->lightingShader->end();
	}

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//frameManager.SwitchToDrawBuffer(prevColorBuffer);

	//glPopAttrib();
}

void GraphicsModel::transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *model)
{
}

std::shared_ptr<ObjectRenderParams> GraphicsModel::createDefaultObjectRenderParams()
{
	return nullptr;
}

void GraphicsModel::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
}