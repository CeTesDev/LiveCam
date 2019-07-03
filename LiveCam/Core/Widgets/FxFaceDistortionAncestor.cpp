#include <Widgets/FxWidgetFaceDistortion.h>

FaceDistortionParams::FaceDistortionParams() { }

void FaceDistortionParams::applySuit(boost::property_tree::ptree& suit, bool loadTexturesImmediately)
{
	
}

static cv::Mat gtexture;
static std::vector<int> triangles;
static std::vector<cv::Point> texture_points;

void FaceDistortionParams::load()
{
}

void FaceDistortion::load()
{

}

void FaceDistortion::transform(ImagePoints &points, FaceDistortionParams &params)
{

}

void FaceDistortion::draw(FaceDistortionParams &params, std::shared_ptr<cwc::glShader> shader)
{

}

void FaceDistortion::unload()
{
	glDeleteBuffers(1, &vbo);

	glDeleteBuffers(1, &texCoordVBO);

	glDeleteBuffers(1, &textureid);
	textureid = 0;
}

GraphicsFaceDistortionModel::GraphicsFaceDistortionModel()
{
}

void GraphicsFaceDistortionModel::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);
}

void GraphicsFaceDistortionModel::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
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

	faceParams[targetIndex].applySuit(suit, loadTexturesImmediately);
}

bool GraphicsFaceDistortionModel::load()
{
	for (auto &params : faceParams)
	{
		params.load();
		break;
	}

	facedistortion.load();

	return true;
}

void GraphicsFaceDistortionModel::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double width = externalRenderParams.ULSWidth;
	double height = externalRenderParams.ULSHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	double h = 30;

	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = 2 / width;
	projection(1, 1) = -2 / height;
	projection(2, 2) = 2 / (zNear - zFar);
	projection(3, 3) = 1;
	projection(0, 3) = -1;
	projection(1, 3) = 1;
	projection(2, 3) = (zFar + zNear) / (zNear - zFar);

	renderParams.projection = projection;

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double planeH = 2.f * h * tan(vAngle / 2);
	double planeW = planeH * aspect;

	const double widthToDepth = 1.0;
	double faceWidth = planeW * widthRawSmooth / face.objWidth;
	double faceDepth = widthToDepth * faceWidth;

	Matrix4d modelMatrix(Affine3d(Translation3d(Vector3d(0, 0, -h - faceDepth))).matrix());

	depth = -h - faceDepth;

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = Matrix3f::Identity();
	renderParams.yawMatrix = Matrix3f::Identity();

	facedistortion.transform(ImagePoints(face.pts), faceParams[face.pointId]);
}

void GraphicsFaceDistortionModel::draw(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	auto shader = objectRenderParams[face.pointId][0]->shader;

	ShaderSetter shaderSetter(shader, *this);

	if (textureid == 0)
	{
		ResourceManager::TextureLoadingResult result = resourceManager.loadTexture(std::string("Texture"), gtexture);

		textureid = result.ID;
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureid);
	facedistortion.textureid = textureid;

	SetUniformsForObject(*shader, 0, face.pointId);

	facedistortion.draw(faceParams[face.pointId], shader);
}

void GraphicsFaceDistortionModel::unload()
{
	facedistortion.unload();
}