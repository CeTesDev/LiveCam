#include <Widgets/FxWidget2D.h>
#include <Graphics/GraphicsLoader.h>

using namespace Eigen;

const std::string FxWidget2D::TYPE_NAME = "2DModel";

std::string FxWidget2D::getTypeName()
{
	return TYPE_NAME;
}

FxWidget2D::FxWidget2D()
{
	emojisOffset.fill({ 0, 0, 0 });
	emojisScale.fill({ 1, 1 });
	emojisPivot.fill({ 0.f, 0.f });

	symmetricState.fill(0);
	emojiRotation.fill(false);
	emojiRolling.fill(true);
	match3D.fill(false);
}

void FxWidget2D::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);
}

void FxWidget2D::prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData)
{
	int renderIDs = findMatchingRenderParamsIndexes
		(modelData.root->filter->commonRenderParams, modelData).front().second.get_value<int>();

	suit.put("renderParamsID", renderIDs);

	auto &emojiTexture = suit.get<std::string>("emojiTexture", "");
	if (!emojiTexture.empty())
	{
		boost::filesystem::path path(emojiTexture);
		suit.put("emojiTexture", path.filename().string());

		auto originFile = fs::path(modelData.root->resourcesRoot / path);
		auto copiedFile = fs::path(modelData.root->filterFolder / path.filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}
}

boost::property_tree::ptree FxWidget2D::getPTree(ExtraModelData &data)
{
	auto tree = GraphicsModel::getPTree(data);

	tree.put("type", getTypeName());

	return tree;
}

void FxWidget2D::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	auto renderParamsID = suit.get_optional<size_t>("renderParamsID");
	if (renderParamsID)
	{
		objectRenderParams[targetIndex].clear();
		objectRenderParams[targetIndex].push_back(commonRenderParams[renderParamsID.get()]);
	}

	symmetricState[targetIndex] = suit.get<int>("symmetricState", 0);

	emojisTextures[targetIndex] = suit.get<std::string>("emojiTexture", "");

	if (loadTexturesImmediately)
	{
		auto loadingResult = resourceManager.loadTexture(emojisTextures[targetIndex]);
		emojisTexturesIDs[targetIndex] = loadingResult.ID;
		textureSize[targetIndex] = { loadingResult.textureWidth, loadingResult.textureHeight };
	}

	emojiRotation[targetIndex] = suit.get<bool>("emojiRotation", false);
	emojiRolling[targetIndex] = suit.get<bool>("emojiRolling", true);

	match3D[targetIndex] = suit.get<bool>("match3D", false);

	auto tree = suit.get_child_optional("emojiOffset");
	emojisOffset[targetIndex] = tree ? JSONVectorReader::readVector3f(tree.get()) : Eigen::Vector3f(0, 0, 0);

	tree = suit.get_child_optional("emojiScale");
	emojisScale[targetIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(1, 1);

	tree = suit.get_child_optional("emojiPivot");
	emojisPivot[targetIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(0, 0);
}

bool FxWidget2D::load()
{
	objects.push_back(GraphicsLoader::CreateQuadModel());

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		auto loadingResult = resourceManager.loadTexture(emojisTextures[i]);
		emojisTexturesIDs[i] = loadingResult.ID;
		textureSize[i] = { loadingResult.textureWidth, loadingResult.textureHeight };
	}

	return true;
}

void FxWidget2D::transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;
	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);
	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = 1;
	projection(1, 1) = 1;
	projection(3, 3) = 1;

	if (match3D[face.pointId])
	{
		projection(2, 2) = (zFar + zNear) / (zNear - zFar);
		projection(2, 3) = 2 * zFar * zNear / (zNear - zFar);
	}
	else
	{
		projection(2, 2) = 2 / (zNear - zFar);
		projection(2, 3) = (zFar + zNear) / (zNear - zFar);
	}

	renderParams.projection = projection;

	auto xRawSmooth = renderParams.xCenterSmoother[face.pointId].smooth(face.xCenterRaw);
	auto yRawSmooth = renderParams.yCenterSmoother[face.pointId].smooth(face.yCenterRaw);

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double rollSmooth, pitchSmooth, yawSmooth;

	pitchSmooth = renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
	yawSmooth = renderParams.yawSmoother[face.pointId].smooth(face.yaw);
	rollSmooth = renderParams.rollSmoother[face.pointId].smooth(face.roll);

	double objWidth = widthRawSmooth / face.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix;
	modelMatrix.setIdentity();

	double xShift = ((xRawSmooth / externalRenderParams.trackedWidth) - 0.5) * 2;
	double yShift = -((yRawSmooth / externalRenderParams.trackedHeight) - 0.5) * 2;

	Affine3d rotation;
	Affine3d anti_rotation = Affine3d::Identity();

	rotation = Affine3d(AngleAxisd(-pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
	rotation *= AngleAxisd(-yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	if (emojiRotation[face.pointId])
	{
		renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
	}
	else
	{
		renderParams.yawMatrix = Matrix3f::Identity();
		anti_rotation = Affine3d(AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
		anti_rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	}

	if (!emojiRotation[face.pointId])
	{
		anti_rotation *= AngleAxisd(-rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));
	}

	bool XMirror = symmetricState[face.pointId] != 0 && -symmetricState[face.pointId] == ((0 < yawSmooth) - (yawSmooth < 0));

	modelMatrix *= Affine3d(Translation3d(Vector3d(xShift, yShift, -distance))).matrix();

	modelMatrix *= Affine3d(Scaling(1., 1. * width / height, 1.)).matrix();

	modelMatrix *= rotation.matrix();

	double Xoffset = (XMirror ? -1.0 : 1.0) * emojisOffset[face.pointId][0] * objWidth;
	double Yoffset = emojisOffset[face.pointId][1] * objWidth;
	double Zoffset = -emojisOffset[face.pointId][2] * STD_HEAD_LENGTH;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, Zoffset))).matrix();

	modelMatrix *= anti_rotation.matrix();

	double Xscale = (XMirror ? -1.0 : 1.0) * emojisScale[face.pointId][0];
	double Yscale = -emojisScale[face.pointId][1] * textureSize[face.pointId][1] / textureSize[face.pointId][0];

	Xoffset = emojisPivot[face.pointId][0] * objWidth * Xscale;
	Yoffset = -emojisPivot[face.pointId][1] * objWidth * Yscale;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, 0))).matrix();

	modelMatrix *= Affine3d(Scaling(objWidth, objWidth, 0.5)).matrix();
	modelMatrix *= Affine3d(Scaling(Xscale, Yscale, 1.)).matrix();

	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	
	renderParams.modelView = modelMatrix.cast<float>();
}

void FxWidget2D::draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	GLuint texID = emojisTexturesIDs[face.pointId];
	if (texID <= 0) return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texID);

	auto shader = objectRenderParams[face.pointId][0]->shader;

	ShaderSetter shaderSetter(shader, *this);

	SetUniformsForObject(*shader, 0, face.pointId);

	VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, objects[0].vb));
	VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*shader, objects[0].tb));

	glDrawArrays(GL_TRIANGLES, 0, 6);

	glPopAttrib();
}

std::shared_ptr<ObjectRenderParams> FxWidget2D::createDefaultObjectRenderParams()
{
	auto result = std::make_shared<ObjectRenderParams>();

	result->shadersSources.first = "./Assets/shaders/vertex/transparent.vertex";
	result->shadersSources.second = "./Assets/shaders/fragment/default.frag";

	auto shader2D = shaderManagerWrapper.LoadFromFile(result->shadersSources.first.data(), result->shadersSources.second.data());
	result->shader = shader2D;
	result->blend = true;
	result->cullFace = false;
	result->alphaTest = true;

	return result;
}

void FxWidget2D::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
	params.blend = true;
	params.cullFace = false;
}

const std::string Suit2DModel::TYPE_NAME = "Suit2DModel";

std::string Suit2DModel::getTypeName()
{
	return TYPE_NAME;
}

void Suit2DModel::transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;
	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);
	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	Matrix4f projection = Matrix4f::Zero();
	projection(0, 0) = 1;
	projection(1, 1) = 1;
	projection(2, 2) = 2 / (zNear - zFar);
	projection(3, 3) = 1;
	projection(0, 3) = 0;
	projection(1, 3) = 0;
	projection(2, 3) = (zFar + zNear) / (zNear - zFar);
	renderParams.projection = projection;

	auto xRawSmooth = renderParams.xCenterSmoother[face.pointId].smooth(face.xCenterRaw);
	auto yRawSmooth = renderParams.yCenterSmoother[face.pointId].smooth(face.yCenterRaw);

	auto rollSmooth = renderParams.rollSmoother[face.pointId].smooth(face.roll);
	auto pitchSmooth = renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
	auto yawSmooth = renderParams.yawSmoother[face.pointId].smooth(face.yaw);

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double objWidth = widthRawSmooth / face.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix;
	modelMatrix.setIdentity();

	double xShift = ((xRawSmooth / externalRenderParams.trackedWidth) - 0.5) * 2;
	double yShift = -((yRawSmooth / externalRenderParams.trackedHeight) - 0.5) * 2;

	renderParams.yawMatrix = Matrix3f::Identity();

	modelMatrix *= Affine3d(Translation3d(Vector3d(xShift, yShift, -distance))).matrix();

	modelMatrix *= Affine3d(Scaling(1., 1. * width / height, 1.)).matrix();

	modelMatrix *= Affine3d(Scaling((double)objWidth, (double)objWidth, 0.5)).matrix();

	double x = sin(yawSmooth / 180.f * M_PI);
	double y = sin(pitchSmooth / 180.f * M_PI);
	modelMatrix *= Affine3d(Translation3d(Eigen::Vector3d(-x, y, 0))).matrix();
	double Xoffset = emojisOffset[face.pointId][0];
	double Yoffset = emojisOffset[face.pointId][1];
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, -emojisOffset[face.pointId][2]))).matrix();

	double Xscale = emojisScale[face.pointId][0];
	double Yscale = -emojisScale[face.pointId][1] * textureSize[face.pointId][1] / textureSize[face.pointId][0];
	modelMatrix *= Affine3d(Scaling(Xscale, Yscale, 1.)).matrix();

	renderParams.rotationMatrix = Eigen::Matrix3f::Identity();

	renderParams.modelView = modelMatrix.cast<float>();
	
}