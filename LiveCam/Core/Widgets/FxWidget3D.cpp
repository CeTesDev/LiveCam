#include <Widgets/FxWidget3D.h>
#include <Graphics/GraphicsLoader.h>
#include <QDebug>

const std::string FxWidget3D::TYPE_NAME = "3DModel";

std::string FxWidget3D::getTypeName()
{
	return TYPE_NAME;
}

void FxWidget3D::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		pivotX[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		pivotY[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
	}
}

FxWidget3D::FxWidget3D()
{
	bmin = { 0, 0, 0 };
	bmax = { 0, 0, 0 };

	backClipping = false;

	pivotOffset = { 0, 0, 0 };
}

boost::property_tree::ptree FxWidget3D::getPTree(ExtraModelData &data)
{
	auto tree = GraphicsModel::getPTree(data);

	tree.put("type", getTypeName());

	if (!modelPath.empty())
	{
		auto originFile = fs::path(data.root->resourcesRoot / modelPath);
		auto copiedFile = fs::path(data.root->filterFolder / fs::path(modelPath).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}

		modelPath = fs::path(modelPath).filename().string();
		tree.put("OBJ", modelPath);
	}

	if (!modelList.empty())
	{
		boost::property_tree::ptree OBJs;

		for (auto &file : modelList)
		{

			auto originFile = fs::path(data.root->resourcesRoot / file);
			auto copiedFile = fs::path(data.root->filterFolder / fs::path(file).filename());
			if (!equivalent(originFile, copiedFile))
			{
				fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
			}

			file = fs::path(file).filename().string();

			boost::property_tree::ptree fileRecord;
			fileRecord.put("", file);
			OBJs.push_back(std::make_pair("", fileRecord));
		}

		tree.put_child("OBJs", OBJs);
	}

	if (!materialPath.empty())
	{
		auto originFile = fs::path(data.root->resourcesRoot / materialPath);
		auto copiedFile = fs::path(data.root->filterFolder / fs::path(materialPath).filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}

		materialPath = fs::path(materialPath).filename().string();
		tree.put("MTL", materialPath);
	}

	if (!materialList.empty())
	{
		boost::property_tree::ptree MTLs;

		for (auto &file : materialList)
		{

			auto originFile = fs::path(data.root->resourcesRoot / file);
			auto copiedFile = fs::path(data.root->filterFolder / fs::path(file).filename());
			if (!equivalent(originFile, copiedFile))
			{
				fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
			}

			file = fs::path(file).filename().string();

			boost::property_tree::ptree fileRecord;
			fileRecord.put("", file);
			MTLs.push_back(std::make_pair("", fileRecord));
		}

		tree.put_child("MTLs", MTLs);
	}

	if (modelShift != Vector3f(0, 0, 0))
	{
		boost::property_tree::ptree shiftTree;

		for (int i = 0; i < 3; ++i)
		{
			boost::property_tree::ptree childTree;
			childTree.put("", modelShift[i]);

			shiftTree.push_back(std::make_pair("", childTree));
		}

		tree.put_child("modelShift", shiftTree);
	}

	if (modelScale != 1)
	{
		tree.put("modelScale", modelScale);
	}

	return tree;
}

void FxWidget3D::prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData)
{
	auto renderIDs = findMatchingRenderParamsIndexes(modelData.root->filter->commonRenderParams, modelData);

	suit.put_child("renderParamsIDs", renderIDs);

	boost::property_tree::ptree modelTexturesTree;
	auto &modelTextures = suit.get_child_optional("modelTextures");
	if (modelTextures)
	{
		for (auto &texture : modelTextures.get())
		{
			boost::filesystem::path path(texture.second.get_value<std::string>());

			boost::property_tree::ptree tree;
			tree.put("", path.filename().string());
			modelTexturesTree.push_back(std::make_pair("", tree));

			if (!path.empty())
			{
				auto originFile = fs::path(modelData.root->resourcesRoot / path);
				auto copiedFile = fs::path(modelData.root->filterFolder / path.filename());
				if (!equivalent(originFile, copiedFile))
				{
					fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
				}
			}
		}

		suit.put_child("modelTextures", modelTexturesTree);
	}
}

void FxWidget3D::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);

	backClipping = modelRecord.get<bool>("backClipping", false);
	Xclip = modelRecord.get<float>("Xclip", std::numeric_limits<float>::max());
	Yclip = modelRecord.get<float>("Yclip", 0);
	Zclip = modelRecord.get<float>("Zclip", 0);

	auto extraShiftVector = modelRecord.get_child_optional("modelShift");
	modelShift = extraShiftVector ? JSONVectorReader::readVector3f(extraShiftVector.get()) : Vector3f(0, 0, 0);

	modelScale = modelRecord.get<float>("modelScale", 1);

	auto rotateMatrixRecord = modelRecord.get_child_optional("modelRotation");
	extraRotateMatrix = rotateMatrixRecord ? JSONVectorReader::readMatrix3f(rotateMatrixRecord.get()) : Matrix3f::Identity();

	modelPath = modelRecord.get<std::string>("OBJ", "");

	materialPath = modelRecord.get<std::string>("MTL", "");

	auto OBJs = modelRecord.get_child_optional("OBJs");
	if (OBJs)
	{
		for (auto &record : OBJs.get())
		{
			modelList.push_back(record.second.get_value<std::string>());
		}
	}

	auto MTLs = modelRecord.get_child_optional("MTLs");
	if (MTLs)
	{
		for (auto &record : MTLs.get())
		{
			materialList.push_back(record.second.get_value<std::string>());
		}
	}
}

bool FxWidget3D::reloadOBJ()
{
	objects.clear();

	float extraScale = 1;
	Vector3f extraShift = Vector3f(0, 0, 0);

	std::vector<std::string> OBJs;

	if (modelList.empty())
	{
		OBJs.push_back((resourceManager.resourceRoot / modelPath).string());
	}
	else
	{
		for (auto &file : modelList)
		{
			OBJs.push_back((resourceManager.resourceRoot / file).string());
		}
	}

	GraphicsLoader::LoadObj(OBJs, objects, bmin, bmax, extraScale, extraShift, extraRotateMatrix);

	GraphicsLoader::loadMTLnames(OBJs, objects);
	
	return true;
}

void FxWidget3D::sortMeshesInZAscending()
{
	std::vector<float> objectsDepths;
	std::vector<int> indexes;

	int i = 0;
	for (auto &obj : objects)
	{
		objectsDepths.push_back(obj.minimumZ);
		indexes.push_back(i++);
	}

	std::sort(indexes.begin(), indexes.end(), [objectsDepths](const int A, const int B) -> bool
	{
		return objectsDepths[A] < objectsDepths[B];
	});

	std::vector<GraphicsData> sortedObjects;

	for (int index : indexes)
	{
		sortedObjects.push_back(objects[index]);
	}

	objects = sortedObjects;
}

void FxWidget3D::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	auto IDs = suit.get_child_optional("renderParamsIDs");
	if (IDs)
	{
		objectRenderParams[targetIndex].clear();
		for (auto &ID : IDs.get())
		{
			objectRenderParams[targetIndex].push_back(commonRenderParams[ID.second.get_value<size_t>()]);
		}
	}

	auto modelTextures = suit.get_child_optional("modelTextures");
	if (modelTextures)
	{
		texturesPaths[targetIndex].clear();

		for (auto &texture : modelTextures.get())
		{
			texturesPaths[targetIndex].push_back(texture.second.get_value<std::string>());
		}

		texturesIDs[targetIndex].clear();

		if (loadTexturesImmediately)
		{
			for (auto &texture : texturesPaths[targetIndex])
			{
				texturesIDs[targetIndex].push_back(resourceManager.loadTexture(texture).ID);
			}
		
			for (auto &params : objectRenderParams[targetIndex])
			{
				if (!params->normalMap.empty())
				{
					params->normalMapID = resourceManager.loadTexture(params->normalMap).ID;
				}
				if (!params->diffuseTexture.empty())
				{
					params->diffuseTextureID = resourceManager.loadTexture(params->diffuseTexture).ID;
				}
				if (!params->specularTexture.empty())
				{
					params->specularTextureID = resourceManager.loadTexture(params->specularTexture).ID;
				}
				if (!params->emissiveTexture.empty())
				{
					params->emissiveTextureID = resourceManager.loadTexture(params->emissiveTexture).ID;
				}
			}
		}
	}
}

bool FxWidget3D::load()
{
	GraphicsModel::load();

	if  (!reloadOBJ())
	{
		return false;
	}

	float width = bmax[0] - bmin[0];
	float height = bmax[1] - bmin[1];
	float depth = bmax[2] - bmin[2];

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		for (auto &particle : particles[i])
		{
			particle.autoScale = width / particle.effect.getEffectWidth() * particle.scale;
			particle.autoGlobalGravity = particle.globalGravity * height;
			particle.autoCoords =
			{ bmin[0] + particle.coords[0] * width, bmin[1] + particle.coords[1] * height, bmin[2] + particle.coords[2] * depth };
			particle.effect.setCoords(particle.autoCoords.head<3>());
		}
	}

	for (int face = 0; face < MAX_TO_TRACK; ++face)
	{
		texturesIDs[face].clear();
		
		for (auto &texture : texturesPaths[face])
		{
			texturesIDs[face].push_back(resourceManager.loadTexture(texture).ID);

			// qDebug() << "FxWidget3D::load: " << texturesIDs[face];
		}

		for (auto &params : objectRenderParams[face])
		{
			if (!params->normalMap.empty())
			{
				params->normalMapID = resourceManager.loadTexture(params->normalMap).ID;
			}
			if (!params->diffuseTexture.empty())
			{
				params->diffuseTextureID = resourceManager.loadTexture(params->diffuseTexture).ID;
			}
			if (!params->specularTexture.empty())
			{
				params->specularTextureID = resourceManager.loadTexture(params->specularTexture).ID;
			}
			if (!params->emissiveTexture.empty())
			{
				params->emissiveTextureID = resourceManager.loadTexture(params->emissiveTexture).ID;
			}
		}
	}

	return true;
}

void FxWidget3D::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	auto xCenterRawSmooth = renderParams.xCenterSmoother[target.pointId].smooth(target.xCenterRaw);
	auto yCenterRawSmooth = renderParams.yCenterSmoother[target.pointId].smooth(target.yCenterRaw);

	auto rollSmooth = renderParams.rollSmoother[target.pointId].smooth(target.roll);
	auto pitchSmooth = renderParams.pitchSmoother[target.pointId].smooth(target.pitch);
	auto yawSmooth = renderParams.yawSmoother[target.pointId].smooth(target.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[target.pointId].smooth(target.widthRaw);

	double width = widthRawSmooth / target.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double tx = xCenterRawSmooth / target.frameWidth;
	double ty = yCenterRawSmooth / target.frameHeight;

	Matrix4d modelMatrix;
	Matrix3f anti_effectRotation;

	double distance = STD_FACE_WIDTH / (width / STD_FACE_DISTANCE);
	depth = -distance;

	double planeH = 2.f * distance * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double xShift = (tx - 0.5) * planeW;
	double yShift = -(ty - 0.5) * planeH;

	Vector3d shift(xShift, yShift, -distance);
	modelMatrix = Affine3d(Translation3d(shift)).matrix();

	Vector3d rotateVector = Vector3d(0.0, 0.0, -1).cross(shift);
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xShift * xShift + yShift * yShift)) / distance);

	Matrix3d correction(AngleAxisd(rotateAngle, rotateVector));
	Matrix3d rotation (AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
	rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0)).matrix();
	rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1)).matrix();

	modelMatrix *= Affine3d(correction * rotation).matrix();

	anti_effectRotation = rotation.inverse().cast<float>();

	shift = { modelShift[0] * STD_FACE_WIDTH,  modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
	modelMatrix *= Affine3d(Translation3d(Vector3d(shift[0], shift[1], shift[2] - STD_HEAD_LENGTH / 2))).matrix();

	double scale = width / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	Matrix4d headAdjusting = Affine3d(Translation3d(autoShift)).matrix();

	modelMatrix *= headAdjusting;

	renderParams.additionalMatrices4[0] = headAdjusting.cast<float>();

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();

	for (auto &particle : particles[target.pointId])
	{
		ParticleUpdateParams params;

		params.outer_transformation = renderParams.modelView;
		params.effect_scaling = Affine3f(Scaling(particle.autoScale)).matrix();
		params.effect_rotation = anti_effectRotation;
		params.globalGravity = particle.autoGlobalGravity;
		params.enable_effect_rotation = particle.enable_rotation;
		params.enable_effect_gravity = particle.enable_effect_gravity;
		params.enable_global_gravity = particle.enable_global_gravity;

		Vector3f newCoords = particle.autoCoords.head<3>();
		if (!particle.enable_rotation)
		{
			newCoords = anti_effectRotation * newCoords;
			particle.effect.setCoords(newCoords);
		}

		newCoords = (params.outer_transformation * Vector4f(newCoords[0], newCoords[1], newCoords[2], 1)).head<3>();

		Vector4f globalSpeed;
		globalSpeed << newCoords - particle.lastCoords, 1;
		Vector3f localSpeed = (params.outer_transformation.inverse() * globalSpeed).head<3>();

		particle.effect.setSpeed({ 0, 0, 0 });

		particle.lastCoords = newCoords;

		particle.effect.update(externalRenderParams.timeShift, params);
	}
	renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}

void FxWidget3D::draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	for (size_t i = 0; i < objects.size(); i++)
	{
		GraphicsData o = objects[i];
		if (o.vb < 1)
		{
			continue;
		}

		int index = i < objectRenderParams[target.pointId].size() ? i : objectRenderParams[target.pointId].size() - 1;
		ObjectRenderParams *renderParams = objectRenderParams[target.pointId][index].get();

		if (!renderParams->visible)
		{
			continue;
		}

		GLuint texId = i < texturesIDs[target.pointId].size() ? texturesIDs[target.pointId][i] :
			texturesIDs[target.pointId].empty() ? 0 : texturesIDs[target.pointId].back();

		if (animateTextures[target.pointId]) {

			if (animateTexturesSpeed[target.pointId]==0) {

				if (animateTexturesCount[target.pointId] < texturesIDs[target.pointId].size() - 1) {

					animateTexturesCount[target.pointId]++;
				}
				else {

					animateTexturesCount[target.pointId] = 0;
				}

			}

			animateTexturesSpeed[target.pointId]++;

			if (animateTexturesSpeed[target.pointId]==3) {
				animateTexturesSpeed[target.pointId] = 0;
			}

			texId = texturesIDs[target.pointId][animateTexturesCount[target.pointId]];
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texId);

		auto shader = renderParams->shader;

		if (renderParams->normalMapID != 0)
		{
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, renderParams->normalMapID);
		}

		ShaderSetter shaderSetter(shader, *this);

		SetUniformsForObject(*shader, i, target.pointId);

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, o.vb));
		VertexAttribSetter vNormal(VertexAttribSetter::NormalAttribSetter(*shader, o.nb));
		VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*shader, o.tb));
		VertexAttribSetter vTangents(VertexAttribSetter::TangentAttribSetter(*shader, o.tangentb));
		VertexAttribSetter vBitangents(VertexAttribSetter::BitangentAttribSetter(*shader, o.bitangentb));

		glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
	}

	// postprocess(target);

	for (auto &particle : particles[target.pointId])
	{
		Matrix4f projection = renderParams.projection;
		particle.effect.draw(projection);
	}

	glPopAttrib();
}

void FxWidget3D::unload()
{
	GraphicsModel::unload();

	for (auto &IDs : texturesIDs)
	{
		IDs.clear();
	}
}

std::shared_ptr<ObjectRenderParams> FxWidget3D::createDefaultObjectRenderParams()
{
	auto result = std::make_shared<ObjectRenderParams>();

	result->shadersSources.first = "./assets/shaders/vertex/notexture.vertex";
	result->shadersSources.second = "./assets/shaders/fragment/notexture.frag";

	auto shader3D = shaderManagerWrapper.LoadFromFile(result->shadersSources.first.data(), result->shadersSources.second.data());
	result->shader = shader3D;
	result->additionalUniforms["cameraPos"] = TUniform4f(0, 0, 0, 1);
	result->additionalUniforms["lightPos"] = TUniform4f(0.2f, 0.2f, 1, 0);
	result->additionalUniforms["materialColor"] = TUniform4f(1, 1, 1, 1);
	result->additionalUniforms["ambientLight"] = TUniform3f(1, 1, 1);
	result->additionalUniforms["diffuseLight"] = TUniform3f(0, 0, 0);
	result->additionalUniforms["bloomLight"] = TUniform3f(0, 0, 0);
	result->additionalUniforms["bloomPower"] = TUniform1f(1);
	result->additionalUniforms["specularLight"] = TUniform3f(0, 0, 0);
	result->additionalUniforms["specularPower"] = TUniform1f(1);
	//result->additionalUniforms["reflectionRatio"] = TUniform1f(0);

	return result;
}

void FxWidget3D::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
	params.additionalUniforms["cameraPos"] = TUniform4f(0, 0, 0, 1);
	params.additionalUniforms["lightPos"] = TUniform4f(0.2f, 0.2f, 1, 0);
	params.additionalUniforms["materialColor"] = TUniform4f(1, 1, 1, 1);
	params.additionalUniforms["ambientLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["diffuseLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["bloomLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["bloomPower"] = TUniform1f(1);
	params.additionalUniforms["specularLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["specularPower"] = TUniform1f(1);
	//params.additionalUniforms["reflectionRatio"] = TUniform1f(0);
}

const std::string LipsJoint3DModel::TYPE_NAME = "LipsJoint3DModel";

std::string LipsJoint3DModel::getTypeName()
{
	return TYPE_NAME;
}

LipsJoint3DModel::LipsJoint3DModel()
{
	pivotOffset = { 0, -0.06, 0 };
}

void LipsJoint3DModel::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	auto xLipsCenter = renderParams.xCenterSmoother[target.pointId].smooth(target.pts[102]);
	auto yLipsCenter = renderParams.yCenterSmoother[target.pointId].smooth(target.pts[103]);
	auto rollSmooth = renderParams.rollSmoother[target.pointId].smooth(target.roll);
	auto pitchSmooth = renderParams.pitchSmoother[target.pointId].smooth(target.pitch);
	auto yawSmooth = renderParams.yawSmoother[target.pointId].smooth(target.yaw);
	auto widthRawSmooth = renderParams.widthSmoother[target.pointId].smooth(target.widthRaw);

	double width = widthRawSmooth / target.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double tx = xLipsCenter / target.frameWidth;
	double ty = yLipsCenter / target.frameHeight;

	Matrix4d modelMatrix;

	AngleAxisd rotationX(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	AngleAxisd rotationY(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	AngleAxisd rotationZ(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	Eigen::Vector3d rotatedPivot = rotationZ * rotationY * rotationX * pivotOffset;

	double distance = STD_FACE_WIDTH / (width / STD_FACE_DISTANCE) - rotatedPivot[2];

	double planeH = 2.f * distance * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double xShift = (tx - 0.5) * planeW;
	double yShift = -(ty - 0.5) * planeH;

	Vector3d shift(xShift, yShift, -distance);
	modelMatrix = Affine3d(Translation3d(shift)).matrix();

	Vector3d rotateVector = Vector3d(0.0, 0.0, -1).cross(shift);
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xShift * xShift + yShift * yShift)) / distance);

	Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
	rotation *= rotationX;
	rotation *= rotationY;
	rotation *= rotationZ;

	modelMatrix *= rotation.matrix();

	shift = { modelShift[0] * STD_FACE_WIDTH, modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
	modelMatrix *= Affine3d(Translation3d(shift)).matrix();

	double scale = width / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	modelMatrix *= Affine3d(Translation3d(autoShift)).matrix();

	depth = 0;

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}
LipsBottomJoint3DModel::LipsBottomJoint3DModel()
{
	pivotOffset = { 0, -0.09, 0 };
}

void LipsBottomJoint3DModel::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	auto xLipsCenter = renderParams.xCenterSmoother[target.pointId].smooth(target.pts[114]);
	auto yLipsCenter = renderParams.yCenterSmoother[target.pointId].smooth(target.pts[115]);

	auto rollSmooth = renderParams.rollSmoother[target.pointId].smooth(target.roll);
	auto pitchSmooth = renderParams.pitchSmoother[target.pointId].smooth(target.pitch);
	auto yawSmooth = renderParams.yawSmoother[target.pointId].smooth(target.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[target.pointId].smooth(target.widthRaw);

	double width = widthRawSmooth / target.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double tx = xLipsCenter / target.frameWidth;
	double ty = yLipsCenter / target.frameHeight;

	Matrix4d modelMatrix;

	AngleAxisd rotationX(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	AngleAxisd rotationY(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	AngleAxisd rotationZ(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	Eigen::Vector3d rotatedPivot = rotationZ * rotationY * rotationX * pivotOffset;

	double distance = STD_FACE_WIDTH / (width / STD_FACE_DISTANCE) - rotatedPivot[2];

	double planeH = 2.f * distance * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double xShift = (tx - 0.5) * planeW;
	double yShift = -(ty - 0.5) * planeH;

	Vector3d shift(xShift, yShift, -distance);
	modelMatrix = Affine3d(Translation3d(shift)).matrix();

	Vector3d rotateVector = Vector3d(0.0, 0.0, -1).cross(shift);
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xShift * xShift + yShift * yShift)) / distance);

	Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
	rotation *= rotationX;
	rotation *= rotationY;
	rotation *= rotationZ;

	modelMatrix *= rotation.matrix();

	shift = { modelShift[0] * STD_FACE_WIDTH, modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
	modelMatrix *= Affine3d(Translation3d(shift)).matrix();

	double scale = width / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	modelMatrix *= Affine3d(Translation3d(autoShift)).matrix();

	depth = 0;

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}
LeftBrow3DModel::LeftBrow3DModel()
{
	pivotOffset = { -0.05, 0.05, 0 };

}

void LeftBrow3DModel::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	auto xCenterSmooth = renderParams.xCenterSmoother[target.pointId].smooth(target.xCenterRaw);
	auto yCenterSmooth = renderParams.yCenterSmoother[target.pointId].smooth(target.yCenterRaw);

	auto xLeftBrow = pivotX[target.pointId].smooth(target.pts[38]);
	auto yLeftBrow = pivotY[target.pointId].smooth(target.pts[39]);

	auto rollSmooth = renderParams.rollSmoother[target.pointId].smooth(target.roll);
	auto pitchSmooth = renderParams.pitchSmoother[target.pointId].smooth(target.pitch);
	auto yawSmooth = renderParams.yawSmoother[target.pointId].smooth(target.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[target.pointId].smooth(target.widthRaw);

	double width = widthRawSmooth / target.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	Matrix4d modelMatrix;

	AngleAxisd rotationX(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	AngleAxisd rotationY(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	AngleAxisd rotationZ(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	Eigen::Vector3d rotatedPivot = rotationZ * rotationY * rotationX * pivotOffset;

	double distance = STD_FACE_WIDTH / (width / STD_FACE_DISTANCE) - rotatedPivot[2];

	double planeH = 2.f * distance * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double xShift = (xLeftBrow / target.frameWidth - 0.5) * planeW;
	double yShift = -(yLeftBrow / target.frameHeight - 0.5) * planeH;

	Vector3d shift(xShift, yShift, -distance);
	modelMatrix = Affine3d(Translation3d(shift)).matrix();

	double xPivotShift = (xCenterSmooth / target.frameWidth - 0.5) * planeW;
	double yPivotShift = -(yCenterSmooth / target.frameWidth - 0.5) * planeH;

	Vector3d pivotShift(xPivotShift, yPivotShift, -distance);
	Vector3d rotateVector = Vector3d(0.0, 0.0, -1).cross(pivotShift);
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xPivotShift * xPivotShift + yPivotShift * yPivotShift)) / distance);

	Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
	rotation *= rotationX;
	rotation *= rotationY;
	rotation *= rotationZ;

	modelMatrix *= rotation.matrix();

	shift = { modelShift[0] * STD_FACE_WIDTH, modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
	modelMatrix *= Affine3d(Translation3d(shift)).matrix();

	Eigen::Vector2f browWidth = { target.pts[42] - target.pts[34] , 0  };
	double scale =  width / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	modelMatrix *= Affine3d(Translation3d(autoShift)).matrix();

	depth = 0;

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}
RightBrow3DModel::RightBrow3DModel()
{
	pivotOffset = { 0.05, 0.05, 0 };
}

void RightBrow3DModel::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	auto xCenterSmooth = renderParams.xCenterSmoother[target.pointId].smooth(target.xCenterRaw);
	auto yCenterSmooth = renderParams.yCenterSmoother[target.pointId].smooth(target.yCenterRaw);

	auto xLeftBrow = pivotX[target.pointId].smooth(target.pts[48]);
	auto yLeftBrow = pivotY[target.pointId].smooth(target.pts[49]);

	auto rollSmooth = renderParams.rollSmoother[target.pointId].smooth(target.roll);
	auto pitchSmooth = renderParams.pitchSmoother[target.pointId].smooth(target.pitch);
	auto yawSmooth = renderParams.yawSmoother[target.pointId].smooth(target.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[target.pointId].smooth(target.widthRaw);

	double width = widthRawSmooth / target.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	Matrix4d modelMatrix;

	AngleAxisd rotationX(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	AngleAxisd rotationY(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	AngleAxisd rotationZ(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	Eigen::Vector3d rotatedPivot = rotationZ * rotationY * rotationX * pivotOffset;

	double distance = STD_FACE_WIDTH / (width / STD_FACE_DISTANCE) - rotatedPivot[2];

	double planeH = 2.f * distance * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double xShift = (xLeftBrow / target.frameWidth - 0.5) * planeW;
	double yShift = -(yLeftBrow / target.frameHeight - 0.5) * planeH;

	Vector3d shift(xShift, yShift, -distance);
	modelMatrix = Affine3d(Translation3d(shift)).matrix();

	double xPivotShift = (xCenterSmooth / target.frameWidth - 0.5) * planeW;
	double yPivotShift = -(yCenterSmooth / target.frameWidth - 0.5) * planeH;

	Vector3d pivotShift(xPivotShift, yPivotShift, -distance);
	Vector3d rotateVector = Vector3d(0.0, 0.0, -1).cross(pivotShift);
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xPivotShift * xPivotShift + yPivotShift * yPivotShift)) / distance);

	Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
	rotation *= rotationX;
	rotation *= rotationY;
	rotation *= rotationZ;

	modelMatrix *= rotation.matrix();

	shift = { modelShift[0] * STD_FACE_WIDTH, modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
	modelMatrix *= Affine3d(Translation3d(shift)).matrix();

	Eigen::Vector2f browWidth = { target.pts[44] - target.pts[52] , 0  };
	double scale =  width / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	modelMatrix *= Affine3d(Translation3d(autoShift)).matrix();

	depth = 0;

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}

Static3DModel::Static3DModel()
{
}

void Static3DModel::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	renderParams.projection = projection;

	Matrix4d modelMatrix;

	modelMatrix = Affine3d(Translation3d(Vector3d(modelShift[0], modelShift[1], modelShift[2]))).matrix();
	
	double scale = 1 / (bmax[0] - bmin[0]) * modelScale;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

	Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	modelMatrix *= Affine3d(Translation3d(autoShift)).matrix();

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = Matrix3f::Identity();
	renderParams.yawMatrix = Matrix3f::Identity();
}