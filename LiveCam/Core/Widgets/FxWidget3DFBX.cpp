#include <Widgets/FxWidget3DFBX.h>
#include <Graphics/GraphicsLoader.h>
#include <QDebug>

// angle of rotation for the camera direction
float angle = 0.0f;

// actual vector representing the camera's direction
float lx = 0.0f, lz = -1.0f;

// XZ position of the camera
float x = 0.0f, z = 800.0f;

// the key states. These variables will be zero
//when no key is being presses
float deltaAngle = 0.0f;
float deltaMove = 0;
int xOrigin = -1;

const std::string FxWidget3DFBX::TYPE_NAME = "3DModel";

fbx_loader::FBXLoader* scene;

std::string FxWidget3DFBX::getTypeName()
{
	return TYPE_NAME;
}

void FxWidget3DFBX::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (int i = 0; i < ObjectTracker::MAX_TO_TRACK; ++i)
	{
		pivotX[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_FACTOR_COORDS * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		pivotY[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_FACTOR_COORDS * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
	}
}

FxWidget3DFBX::FxWidget3DFBX()
{
	bmin = { 0, 0, 0 };
	bmax = { 0, 0, 0 };

	backClipping = false;

	pivotOffset = { 0, 0, 0 };
}

boost::property_tree::ptree FxWidget3DFBX::getPTree(ExtraModelData &data)
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

void FxWidget3DFBX::prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData)
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

void FxWidget3DFBX::loadFromJSON(boost::property_tree::ptree& modelRecord)
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
}

bool FxWidget3DFBX::reloadOBJ()
{
	objects.clear();
	GraphicsLoader::LoadFbx((resourceManager.resourceRoot / modelPath).string(), scene);
	
	return true;
}

void FxWidget3DFBX::sortMeshesInZAscending()
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

void FxWidget3DFBX::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
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
			}
		}
	}
}

bool FxWidget3DFBX::load()
{
	GraphicsModel::load();

	if  (!reloadOBJ())
	{
		return false;
	}

	float width = bmax[0] - bmin[0];
	float height = bmax[1] - bmin[1];
	float depth = bmax[2] - bmin[2];

	for (int i = 0; i < ObjectTracker::MAX_TO_TRACK; ++i)
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

	for (int face = 0; face < ObjectTracker::MAX_TO_TRACK; ++face)
	{
		texturesIDs[face].clear();
		
		for (auto &texture : texturesPaths[face])
		{
			texturesIDs[face].push_back(resourceManager.loadTexture(texture).ID);

			qDebug() << "FxWidget3DFBX::load: " << texturesIDs[face];
		}

		for (auto &params : objectRenderParams[face])
		{
			if (!params->normalMap.empty())
			{
				params->normalMapID = resourceManager.loadTexture(params->normalMap).ID;
			}
		}
	}

	return true;
}

void FxWidget3DFBX::transform(FXModel& fxModel, ExternalRenderParams &externalRenderParams)
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

	auto xCenterRawSmooth = renderParams.xCenterSmoother[fxModel.pointId].smooth(fxModel.xCenterRaw);
	auto yCenterRawSmooth = renderParams.yCenterSmoother[fxModel.pointId].smooth(fxModel.yCenterRaw);

	auto rollSmooth = renderParams.rollSmoother[fxModel.pointId].smooth(fxModel.roll);
	auto pitchSmooth = renderParams.pitchSmoother[fxModel.pointId].smooth(fxModel.pitch);
	auto yawSmooth = renderParams.yawSmoother[fxModel.pointId].smooth(fxModel.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[fxModel.pointId].smooth(fxModel.widthRaw);

	double width = widthRawSmooth / fxModel.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double tx = xCenterRawSmooth / fxModel.frameWidth;
	double ty = yCenterRawSmooth / fxModel.frameHeight;

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

	for (auto &particle : particles[fxModel.pointId])
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

void FxWidget3DFBX::draw(FXModel& fxModel, ExternalRenderParams &externalRenderParams)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	// qDebug() << "FxWidget3DFBX::objects.size(): " << objects.size();

	scene->Draw();

	for (auto &particle : particles[fxModel.pointId])
	{
		Matrix4f projection = renderParams.projection;
		particle.effect.draw(projection);
	}

	glPopAttrib();
}

void FxWidget3DFBX::unload()
{
	GraphicsModel::unload();

	for (auto &IDs : texturesIDs)
	{
		IDs.clear();
	}
}

std::shared_ptr<ObjectRenderParams> FxWidget3DFBX::createDefaultObjectRenderParams()
{
	auto result = std::make_shared<ObjectRenderParams>();

	result->shadersSources.first = "./assets/shaders/vertex/noTextureVertexShader.txt";
	result->shadersSources.second = "./assets/shaders/fragment/noTexturePhongFragmentShader.txt";

	auto shader3D = shaderManagerWrapper.LoadFromFile(result->shadersSources.first.data(), result->shadersSources.second.data());
	result->shader = shader3D;
	result->additionalUniforms["cameraPos"] = TUniform4f(0, 0, 0, 1);
	result->additionalUniforms["lightPos"] = TUniform4f(0.2f, 0.2f, 1, 0);
	result->additionalUniforms["materialColor"] = TUniform4f(1, 1, 1, 1);
	result->additionalUniforms["ambientLight"] = TUniform3f(1, 1, 1);
	result->additionalUniforms["diffuseLight"] = TUniform3f(0, 0, 0);
	result->additionalUniforms["specularLight"] = TUniform3f(0, 0, 0);
	result->additionalUniforms["specularPower"] = TUniform1f(1);
	result->additionalUniforms["reflectionRatio"] = TUniform1f(0);

	return result;
}

void FxWidget3DFBX::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
	params.additionalUniforms["cameraPos"] = TUniform4f(0, 0, 0, 1);
	params.additionalUniforms["lightPos"] = TUniform4f(0.2f, 0.2f, 1, 0);
	params.additionalUniforms["materialColor"] = TUniform4f(1, 1, 1, 1);
	params.additionalUniforms["ambientLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["diffuseLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["specularLight"] = TUniform3f(0, 0, 0);
	params.additionalUniforms["specularPower"] = TUniform1f(1);
	params.additionalUniforms["reflectionRatio"] = TUniform1f(0);
}

