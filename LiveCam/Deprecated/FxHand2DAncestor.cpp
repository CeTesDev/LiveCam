#include "fx/FxHand2DAncestor.h"

using namespace Eigen;

const std::string GraphicsHand2DModel::TYPE_NAME = "Hand2DModel";

std::string GraphicsHand2DModel::getTypeName()
{
	return TYPE_NAME;
}

GraphicsHand2DModel::GraphicsHand2DModel()
{
	handsOffset.fill({ 0, 0, 0 });
	handsScale.fill({ 1, 1 });
	handsPivot.fill({ 0.f, 0.f });

	symmetricState.fill(0);
	handRotation.fill(false);
	handRolling.fill(true);
	match3D.fill(false);
}

void GraphicsHand2DModel::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);
}

void GraphicsHand2DModel::prepareSuitForJSON(boost::property_tree::ptree &suit, ExtraModelData& modelData)
{
	int renderIDs = findMatchingRenderParamsIndexes
		(modelData.root->filter->commonRenderParams, modelData).front().second.get_value<int>();

	suit.put("renderParamsID", renderIDs);

	auto &handTexture = suit.get<std::string>("handTexture", "");
	if (!handTexture.empty())
	{
		boost::filesystem::path path(handTexture);
		suit.put("handTexture", path.filename().string());

		auto originFile = fs::path(modelData.root->resourcesRoot / path);
		auto copiedFile = fs::path(modelData.root->filterFolder / path.filename());
		if (!equivalent(originFile, copiedFile))
		{
			fs::copy_file(originFile, copiedFile, fs::copy_option::overwrite_if_exists);
		}
	}
}

boost::property_tree::ptree GraphicsHand2DModel::getPTree(ExtraModelData &data)
{
	auto tree = GraphicsModel::getPTree(data);

	tree.put("type", getTypeName());

	return tree;
}

void GraphicsHand2DModel::applySuit(boost::property_tree::ptree& suit, size_t faceIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, faceIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[faceIndex])
	{
		return;
	}

	auto renderParamsID = suit.get_optional<size_t>("renderParamsID");
	if (renderParamsID)
	{
		objectRenderParams[faceIndex].clear();
		objectRenderParams[faceIndex].push_back(commonRenderParams[renderParamsID.get()]);
	}

	symmetricState[faceIndex] = suit.get<int>("symmetricState", 0);

	handsTextures[faceIndex] = suit.get<std::string>("handTexture", "");

	if (loadTexturesImmediately)
	{
		auto loadingResult = resourceManager.loadTexture(handsTextures[faceIndex]);
		handsTexturesIDs[faceIndex] = loadingResult.ID;
		textureSize[faceIndex] = { loadingResult.textureWidth, loadingResult.textureHeight };
	}

	handRotation[faceIndex] = suit.get<bool>("handRotation", false);
	handRolling[faceIndex] = suit.get<bool>("handRolling", true);

	match3D[faceIndex] = suit.get<bool>("match3D", false);

	auto tree = suit.get_child_optional("handOffset");
	handsOffset[faceIndex] = tree ? JSONVectorReader::readVector3f(tree.get()) : Eigen::Vector3f(0, 0, 0);

	tree = suit.get_child_optional("handScale");
	handsScale[faceIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(1, 1);

	tree = suit.get_child_optional("handPivot");
	handsPivot[faceIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(0, 0);
}

bool GraphicsHand2DModel::load()
{
	objects.push_back(GraphicsLoader::CreateQuadModel());

	for (int i = 0; i < ObjectTracker::MAX_TO_TRACK; ++i)
	{
		auto loadingResult = resourceManager.loadTexture(handsTextures[i]);
		handsTexturesIDs[i] = loadingResult.ID;
		textureSize[i] = { loadingResult.textureWidth, loadingResult.textureHeight };
	}

	return true;
}

void GraphicsHand2DModel::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;
	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);
	double width = externalRenderParams.ULSWidth;
	double height = externalRenderParams.ULSHeight;
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

	auto xRawSmooth = renderParams.faceXCenterSmoother[face.pointId].responsiveAnalogReadSimple(face.xFaceCenterRaw);
	auto yRawSmooth = renderParams.faceYCenterSmoother[face.pointId].responsiveAnalogReadSimple(face.yFaceCenterRaw);

	auto faceWidthRawSmooth = renderParams.faceWidthSmoother[face.pointId].responsiveAnalogReadSimple(face.faceWidthRaw);

	double rollSmooth, pitchSmooth, yawSmooth;

	pitchSmooth = renderParams.pitchSmoother[face.pointId].responsiveAnalogReadSimple(face.pitch);
	yawSmooth = renderParams.yawSmoother[face.pointId].responsiveAnalogReadSimple(face.yaw);
	rollSmooth = renderParams.rollSmoother[face.pointId].responsiveAnalogReadSimple(face.roll);

	double faceWidth = faceWidthRawSmooth / face.fWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (faceWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix;
	modelMatrix.setIdentity();

	double xShift = ((xRawSmooth / externalRenderParams.ULSWidth) - 0.5) * 2;
	double yShift = -((yRawSmooth / externalRenderParams.ULSHeight) - 0.5) * 2;

	Affine3d rotation;
	Affine3d anti_rotation = Affine3d::Identity();

	rotation = Affine3d(AngleAxisd(-pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
	rotation *= AngleAxisd(-yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	if (handRotation[face.pointId])
	{
		renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
	}
	else
	{
		renderParams.yawMatrix = Matrix3f::Identity();
		anti_rotation = Affine3d(AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
		anti_rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	}

	if (!handRotation[face.pointId])
	{
		anti_rotation *= AngleAxisd(-rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));
	}

	bool XMirror = symmetricState[face.pointId] != 0 && -symmetricState[face.pointId] == ((0 < yawSmooth) - (yawSmooth < 0));

	modelMatrix *= Affine3d(Translation3d(Vector3d(xShift, yShift, -distance))).matrix(); // offset to faceCenter

	modelMatrix *= Affine3d(Scaling(1., 1. * width / height, 1.)).matrix(); // screen aspect ratio

	modelMatrix *= rotation.matrix(); // rotation

	// auto topPoint = PointCalcUtil::centerOf2Points({ (int)face.pts[38], (int)face.pts[39] }, { (int)face.pts[48], (int)face.pts[49] });
	// float faceHeight = PointCalcUtil::distanceBetween({ (int)face.pts[16], (int)face.pts[17] }, topPoint);

	double Xoffset = (XMirror ? -1.0 : 1.0) * handsOffset[face.pointId][0] * faceWidth;
	double Yoffset = handsOffset[face.pointId][1] * faceWidth;
	double Zoffset = -handsOffset[face.pointId][2] * STD_HEAD_LENGTH;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, Zoffset))).matrix(); // offset relativly faceWidth

	modelMatrix *= anti_rotation.matrix(); // anti-rotation to prevent yaw & pitch rotation to keep screen-perpendicularly hand orientation

	double Xscale = (XMirror ? -1.0 : 1.0) * handsScale[face.pointId][0];
	double Yscale = -handsScale[face.pointId][1] * textureSize[face.pointId][1] / textureSize[face.pointId][0];

	Xoffset = handsPivot[face.pointId][0] * faceWidth * Xscale;
	Yoffset = -handsPivot[face.pointId][1] * faceWidth * Yscale;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, 0))).matrix(); // go to pivot

	modelMatrix *= Affine3d(Scaling(faceWidth, faceWidth, 0.5)).matrix(); // resize for one atlas texture according to faceWidth
	modelMatrix *= Affine3d(Scaling(Xscale, Yscale, 1.)).matrix(); // head-relative scale & distortion fixing & Y mirror

	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();

	renderParams.modelView = modelMatrix.cast<float>();
}

void GraphicsHand2DModel::draw(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	GLuint texID = handsTexturesIDs[face.pointId];
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

std::shared_ptr<ObjectRenderParams> GraphicsHand2DModel::createDefaultObjectRenderParams()
{
	auto result = std::make_shared<ObjectRenderParams>();

	result->shadersSources.first = "./assets/shaders/vertex/orthoVertexShader.txt";
	result->shadersSources.second = "./assets/shaders/fragment/orthoFragmentShader.txt";

	auto shader2D = shaderManagerWrapper.LoadFromFile(result->shadersSources.first.data(), result->shadersSources.second.data());
	result->shader = shader2D;
	result->blend = true;
	result->cullFace = false;
	result->alphaTest = true;

	return result;
}

void GraphicsHand2DModel::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
	params.blend = true;
	params.cullFace = false;
}


const std::string Hand2DModel::TYPE_NAME = "Hand2DModel";

std::string Hand2DModel::getTypeName()
{
	return TYPE_NAME;
}

void Hand2DModel::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;
	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);
	double width = externalRenderParams.ULSWidth;
	double height = externalRenderParams.ULSHeight;
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

	auto xRawSmooth = renderParams.faceXCenterSmoother[face.pointId].responsiveAnalogReadSimple(face.xFaceCenterRaw);
	auto yRawSmooth = renderParams.faceYCenterSmoother[face.pointId].responsiveAnalogReadSimple(face.yFaceCenterRaw);

	auto rollSmooth = renderParams.rollSmoother[face.pointId].responsiveAnalogReadSimple(face.roll);
	auto pitchSmooth = renderParams.pitchSmoother[face.pointId].responsiveAnalogReadSimple(face.pitch);
	auto yawSmooth = renderParams.yawSmoother[face.pointId].responsiveAnalogReadSimple(face.yaw);

	auto faceWidthRawSmooth = renderParams.faceWidthSmoother[face.pointId].responsiveAnalogReadSimple(face.faceWidthRaw);

	double faceWidth = faceWidthRawSmooth / face.fWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (faceWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix;
	modelMatrix.setIdentity();

	double xShift = ((xRawSmooth / externalRenderParams.ULSWidth) - 0.5) * 2;
	double yShift = -((yRawSmooth / externalRenderParams.ULSHeight) - 0.5) * 2;

	renderParams.yawMatrix = Matrix3f::Identity();

	modelMatrix *= Affine3d(Translation3d(Vector3d(xShift, yShift, -distance))).matrix(); // offset to faceCenter

	modelMatrix *= Affine3d(Scaling(1., 1. * width / height, 1.)).matrix(); // screen aspect ratio

	modelMatrix *= Affine3d(Scaling((double)faceWidth, (double)faceWidth, 0.5)).matrix(); // resize according to faceWidth

	double x = sin(yawSmooth / 180.f * M_PI);
	double y = sin(pitchSmooth / 180.f * M_PI);
	//double z = sqrt(1 - x * x + y * y);
	modelMatrix *= Affine3d(Translation3d(Eigen::Vector3d(-x, y, 0))).matrix(); // head center shift

	// 		auto topPoint = PointCalcUtil::centerOf2Points({ (int)face.pts[38], (int)face.pts[39] }, { (int)face.pts[48], (int)face.pts[49] });
	// 		float faceHeight = PointCalcUtil::distanceBetween({ (int)face.pts[16], (int)face.pts[17] }, topPoint);
	double Xoffset = handsOffset[face.pointId][0];
	double Yoffset = handsOffset[face.pointId][1];
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, -handsOffset[face.pointId][2]))).matrix(); // offset relativly faceWidth

	double Xscale = handsScale[face.pointId][0];
	double Yscale = -handsScale[face.pointId][1] * textureSize[face.pointId][1] / textureSize[face.pointId][0];
	modelMatrix *= Affine3d(Scaling(Xscale, Yscale, 1.)).matrix(); // head-relative scale & distortion fixing & Y mirror

	renderParams.rotationMatrix = Eigen::Matrix3f::Identity();

	renderParams.modelView = modelMatrix.cast<float>();

}
