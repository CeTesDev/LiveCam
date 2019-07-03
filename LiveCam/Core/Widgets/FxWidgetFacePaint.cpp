#include <Widgets/FxWidgetFacePaint.h>

extern GLuint cameraTextureId;

const static cv::Point icon_position(296, 538);

FaceMapping::Face FaceModel;

static cv::Mat gtexture;

//-----------------------------------------------------------------------------

HalfFaceParams::HalfFaceParams() { }

HalfFaceParams::HalfFaceParams(Eigen::Vector4f leftAmbient, Eigen::Vector4f rightAmbient, Eigen::Vector4f lineAmbient,
	Eigen::Vector4f leftDiffuse, Eigen::Vector4f rightDiffuse, Eigen::Vector4f lineDiffuse,
	std::string leftTexture, std::string rightTexture, std::string lineTexture, float lineWidth,
	std::string alphaMask, float alphaMaskPower)
	: leftAmbient(leftAmbient)
	, rightAmbient(rightAmbient)
	, lineAmbient(lineAmbient)
	, leftDiffuse(leftDiffuse)
	, rightDiffuse(rightDiffuse)
	, lineDiffuse(lineDiffuse)
	, leftTexture(leftTexture)
	, rightTexture(rightTexture)
	, lineTexture(lineTexture)
	, lineWidth(lineWidth)
	, alphaMask(alphaMask)
	, alphaMaskPower(alphaMaskPower)
{ }

void HalfFaceParams::applySuit(boost::property_tree::ptree& suit, bool loadTexturesImmediately)
{
	auto tree = suit.get_child_optional("leftAmbient");
	leftAmbient = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	tree = suit.get_child_optional("rightAmbient");
	rightAmbient = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	tree = suit.get_child_optional("lineAmbient");
	lineAmbient = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	tree = suit.get_child_optional("leftDiffuse");
	leftDiffuse = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	tree = suit.get_child_optional("rightDiffuse");
	rightDiffuse = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	tree = suit.get_child_optional("lineDiffuse");
	lineDiffuse = tree ? JSONVectorReader::readVector4f(tree.get()) : Eigen::Vector4f(0, 0, 0, 0);

	leftTexture = suit.get<std::string>("leftTexture", "");
	rightTexture = suit.get<std::string>("rightTexture", "");
	lineTexture = suit.get<std::string>("lineTexture", "");
	alphaMask = suit.get<std::string>("alphaMask", "");

	alphaMaskPower = suit.get<float>("alphaMaskPower", 1);
	lineWidth = suit.get<float>("lineWidth", 0);
	verticalOffset = suit.get<float>("verticalOffset", 0);

	if (loadTexturesImmediately)
	{
		leftTexId = resourceManager.loadTexture(leftTexture).ID;
		rightTexId = resourceManager.loadTexture(rightTexture).ID;
		lineTexId = resourceManager.loadTexture(lineTexture).ID;
		alphaMaskId = resourceManager.loadTexture(alphaMask).ID;
	}
}

void HalfFaceParams::load()
{
	leftTexId = resourceManager.loadTexture(leftTexture).ID;
	auto i = resourceManager.loadTexture(rightTexture);
	GLuint id = i.ID;
	rightTexId = resourceManager.loadTexture(rightTexture).ID;
	lineTexId = resourceManager.loadTexture(lineTexture).ID;
	alphaMaskId = resourceManager.loadTexture(alphaMask).ID;
}

void HalfFacePaint::load(bool headTextureMap)
{
	glGenBuffers(1, &leftVBO);
	glGenBuffers(1, &rightVBO);
	glGenBuffers(1, &lineVBO);

	glGenBuffers(1, &leftTexCoordVBO);
	glGenBuffers(1, &rightTexCoordVBO);
	glGenBuffers(1, &lineTexCoordVBO);

	facePoints = FaceModel.getCoords();

	for (int i = 0; i < 2; ++i)
	{
		if (headTextureMap)
		{
			for (size_t j = 0; j < FaceMapping::TOTAL_VERTEX_POINTS; j++)
			{
				facePoints[i][j] = FaceMapping::TexMapTable[j];
			}
		}
		else
		{
			for (size_t j = 0; j < FaceMapping::TOTAL_VERTEX_POINTS; j++)
			{
				facePoints[i][j].x /= 1000.f;
				facePoints[i][j].y /= 1000.f;
			}
		}

		std::array<float, FaceMapping::TOTAL_VERTEX_POINTS * 2> texCoordBuffer;

		auto iter = texCoordBuffer.begin();
		for (size_t j = 0; j < FaceMapping::TOTAL_VERTEX_POINTS; j++)
		{
			*iter++ = facePoints[i][j].x;
			*iter++ = facePoints[i][j].y;
		}

		if (i == 0)
		{
			glBindBuffer(GL_ARRAY_BUFFER, leftTexCoordVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * FaceMapping::TOTAL_VERTEX_POINTS * 2, texCoordBuffer.data(), GL_STATIC_DRAW);
		}
		else
		{
			glBindBuffer(GL_ARRAY_BUFFER, rightTexCoordVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * FaceMapping::TOTAL_VERTEX_POINTS * 2, texCoordBuffer.data(), GL_STATIC_DRAW);
		}
	}
}

void HalfFacePaint::transform(TrackingTarget& face, HalfFaceParams &params)
{
	std::array<float, TARGET_DETAIL_LIMIT * 2> smoothedPoints;
	for (size_t j = 0; j < TOTAL_DETAIL_POINTS; ++j)
	{
		smoothedPoints[2 * j] = facePointsSmoothers[face.pointId][2 * j].smooth(face.pts[2 * j]);
		smoothedPoints[2 * j + 1] = facePointsSmoothers[face.pointId][2 * j + 1].smooth(face.pts[2 * j + 1]);
	}
	ImagePoints points(smoothedPoints);
	FaceModel.updateCoords(points);
	facePoints = FaceModel.getCoords();

	Eigen::Vector2f topPoint = { FaceModel.top.x, FaceModel.top.y };
	Eigen::Vector2f verticalOffset = points.getEigenAt(28) - Eigen::Vector2f(FaceModel.bottom.x, FaceModel.bottom.y);
	verticalOffset *= ((params.verticalOffset == 0.0) ? 0.2 : params.verticalOffset);

	int index = 0;
	for (int i = 9; i > 0; --i)
	{
		faceLeftContour[index++] = { points.getEigenAt(i) };
	}

	index = 0;
	for (int i = 9; i < 18; ++i)
	{
		faceRightContour[index++] = { points.getEigenAt(i) };
	}

	index = 0;
	for (int i = 37; i < 43; ++i)
	{
		eyeLeftContour[index++] = { points.getEigenAt(i) };
	}
	eyeLeftContour[index] = { points.getEigenAt(37) };

	index = 0;
	for (int i = 43; i < 49; ++i)
	{
		eyeRightContour[index++] = { points.getEigenAt(i) };
	}
	eyeRightContour[index] = { points.getEigenAt(43) };

	index = 0;
	for (int i = 49; i < 55; ++i)
	{
		mouthLeftContour[index++] = { points.getEigenAt(i) };
	}

	index = 0;
	for (int i = 55; i < 61; ++i)
	{
		mouthRightContour[index++] = { points.getEigenAt(i) };
	}

	index = 0;
	middleTopLines[index++] = topPoint + verticalOffset;
	middleTopLines[index++] = topPoint;
	for (int i = 28; i < 32; ++i)
	{
		middleTopLines[index++] = { points.getEigenAt(i) };
	}
	middleTopLines[index++] = { points.getEigenAt(34) };
	middleTopLines[index++] = { points.getEigenAt(52) };

	middleBottomLines[0] = { points.getEigenAt(58) };
	middleBottomLines[1] = { points.getEigenAt(9) };

	for (int i = 0; i < 2; ++i)
	{

		std::array<float, (FaceMapping::TOTAL_VERTEX_POINTS) * 3> buffer;

		auto iter = buffer.begin();
		for (size_t j = 0; j < FaceMapping::TOTAL_VERTEX_POINTS; j++)
		{
			*iter++ = facePoints[i][j].x;
			*iter++ = facePoints[i][j].y;
			*iter++ = 0;
		}
		if (i == 0)
		{
			/*faceLeftContour[9] = { facePoints[i][43].x, facePoints[i][43].y };
			faceLeftContour[10] = { facePoints[i][45].x, facePoints[i][45].y };
			faceLeftContour[11] = { facePoints[i][47].x, facePoints[i][47].y };
			faceLeftContour[12] = { facePoints[i][49].x, facePoints[i][49].y };*/

			glBindBuffer(GL_ARRAY_BUFFER, leftVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * TARGET_DETAIL_LIMIT * 3, buffer.data(), GL_STREAM_DRAW);
		}
		else
		{
			/*faceRightContour[9] = { facePoints[i][43].x, facePoints[i][43].y };
			faceRightContour[10] = { facePoints[i][45].x, facePoints[i][45].y };
			faceRightContour[11] = { facePoints[i][47].x, facePoints[i][47].y };
			faceRightContour[12] = { facePoints[i][49].x, facePoints[i][49].y };*/

			glBindBuffer(GL_ARRAY_BUFFER, rightVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * TARGET_DETAIL_LIMIT * 3, buffer.data(), GL_STREAM_DRAW);
		}
	}

	if ((params.lineAmbient[3] != 0 || params.lineTexId != 0) && params.lineWidth != 0)
	{
		const std::array<Eigen::Vector2f, FaceMapping::LINE_STRIP_COUNT / 2> line_strip =
		{
			topPoint + verticalOffset, topPoint,
			points.getEigenAt(28), points.getEigenAt(29), points.getEigenAt(30), points.getEigenAt(31), points.getEigenAt(34), points.getEigenAt(52)
		};

		std::array<float, (FaceMapping::LINE_STRIP_COUNT + 4) * 3 > buffer;
		auto iter = buffer.begin();

		Eigen::Vector2f route = PointCalcUtil::getVectorNormal(line_strip[1] - line_strip[0]);

		Eigen::Vector2f point = line_strip[0] + route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		 point = line_strip[0] - route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		for (int i = 1; i < FaceMapping::LINE_STRIP_COUNT / 2 - 1; ++i)
		{
			route = PointCalcUtil::getBisector(line_strip[i - 1] - line_strip[i], line_strip[i + 1] - line_strip[i]);
			if (route[0] < 0) route[0] = -route[0];

			point = line_strip[i] + route * params.lineWidth;
			*iter++ = point[0];
			*iter++ = point[1];
			*iter++ = 0;

			point = line_strip[i] - route * params.lineWidth;
			*iter++ = point[0];
			*iter++ = point[1];
			*iter++ = 0;
		}

		route = PointCalcUtil::getVectorNormal(line_strip[FaceMapping::LINE_STRIP_COUNT / 2 - 1] - line_strip[FaceMapping::LINE_STRIP_COUNT / 2 - 2]);
		
		point = line_strip[FaceMapping::LINE_STRIP_COUNT / 2 - 1] + route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		point = line_strip[FaceMapping::LINE_STRIP_COUNT / 2 - 1] - route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		route = PointCalcUtil::getVectorNormal(points.getEigenAt(9) - points.getEigenAt(58));

		point = points.getEigenAt(58) + route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		point = points.getEigenAt(58) - route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		point = points.getEigenAt(9) + route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		point = points.getEigenAt(9) - route * params.lineWidth;
		*iter++ = point[0];
		*iter++ = point[1];
		*iter++ = 0;

		glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (FaceMapping::LINE_STRIP_COUNT + 4) * 3, buffer.data(), GL_STREAM_DRAW);
	}
}

void HalfFacePaint::draw(TrackingTarget& face, HalfFaceParams &params, std::shared_ptr<cwc::glShader> shader)
{

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glDisable(GL_CULL_FACE);

	glEnable(GL_BLEND);

	int texture = shader->GetUniformLocation("Texture");
	glUniform1i(texture, 0);

	int alphaMask = shader->GetUniformLocation("AlphaMaskTexture");
	glUniform1i(alphaMask, 1);

	if (params.leftAmbient[3] != 0)
	{
		shader->setUniform4f("leftColor", params.leftAmbient[0], params.leftAmbient[1], params.leftAmbient[2], params.leftAmbient[3]);
	}
	else
	{
		shader->setUniform4f("leftColor", 0, 0, 0, 0);
	}

	if (params.rightAmbient[3] != 0)
	{
		shader->setUniform4f("rightColor", params.rightAmbient[0], params.rightAmbient[1], params.rightAmbient[2], params.rightAmbient[3]);
	}
	else
	{
		shader->setUniform4f("rightColor", 0, 0, 0, 0);
	}

	GLint edgeFadingDistanceLoc = shader->GetUniformLocation("edgeFadingDistance");
	if (edgeFadingDistanceLoc > -1)
	{
		float edgeFadingDistance = face.width * 0.05f * face.frameWidth;
		shader->setUniform1f("edgeFadingDistance", edgeFadingDistance);
		PointCalcUtil::LoadLinesToUniform(middleTopLines, "edgesMiddleTop", shader);
		PointCalcUtil::LoadLinesToUniform(middleBottomLines, "edgesMiddleBottom", shader);
	}
	{			
		if (edgeFadingDistanceLoc > -1)
		{
			PointCalcUtil::LoadLinesToUniform(faceLeftContour, "edgesLeft", shader);
			PointCalcUtil::LoadLinesToUniform(eyeLeftContour, "edgesLeft", shader);
			PointCalcUtil::LoadLinesToUniform(mouthLeftContour, "edgesLeft", shader);
		}

		shader->setUniform1i("isLeft", 1);

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, leftVBO));
		VertexAttribSetter vTexCoord;
		{
			vTexCoord = VertexAttribSetter(VertexAttribSetter::TexCoordAttribSetter(*shader, leftTexCoordVBO));
		}

		if (params.leftTexId != 0)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, params.leftTexId);

			shader->setUniform1i("hasTexture", 1);
		}
		else
		{
			shader->setUniform1i("hasTexture", 0);
		}

		if (params.alphaMaskId != 0 && params.alphaMaskPower > 0)
		{
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, params.alphaMaskId);

			shader->setUniform1i("hasAlphaMask", 1);
			shader->setUniform1f("alphaMaskPower", params.alphaMaskPower);
		}
		else
		{
			shader->setUniform1i("hasAlphaMask", 0);
		}

		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::CHIN_POINTS_START, FaceMapping::TOTAL_CHIN_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::CHEEK_POINTS_START, FaceMapping::TOTAL_CHEEK_POINTS);
		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::BROW_POINTS_START, FaceMapping::TOTAL_BROW_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::NOSE_POINTS_START, FaceMapping::TOTAL_NOSE_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::PHILTRUM_POINTS_START, FaceMapping::TOTAL_PHILTRUM_POINTS);
		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::FOREHEAD_POINTS_START, FaceMapping::TOTAL_FOREHEAD_POINTS);
		if (params.texturedMouth)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::MOUTH_POINTS_START, FaceMapping::TOTAL_MOUTH_POINTS);
		}
		if (params.texturedEyes)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::EYE_POINTS_START, FaceMapping::TOTAL_EYE_POINTS);
		}
	}
	{
		if (edgeFadingDistanceLoc > -1)
		{
			PointCalcUtil::LoadLinesToUniform(faceRightContour, "edgesRight", shader);
			PointCalcUtil::LoadLinesToUniform(eyeRightContour, "edgesRight", shader);
			PointCalcUtil::LoadLinesToUniform(mouthRightContour, "edgesRight", shader);
		}

		shader->setUniform1i("isLeft", 0);

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, rightVBO));
		VertexAttribSetter vTexCoord;
		{
			vTexCoord = VertexAttribSetter(VertexAttribSetter::TexCoordAttribSetter(*shader, rightTexCoordVBO));
		}

		if (params.rightTexId != 0)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, params.rightTexId);

			shader->setUniform1i("hasTexture", 1);
		}
		else
		{
			shader->setUniform1i("hasTexture", 0);
		}

		if (params.alphaMaskId != 0 && params.alphaMaskPower > 0)
		{
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, params.alphaMaskId);

			shader->setUniform1i("hasAlphaMask", 1);
			shader->setUniform1f("alphaMaskPower", params.alphaMaskPower);
		}
		else
		{
			shader->setUniform1i("hasAlphaMask", 0);
		}

		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::CHIN_POINTS_START, FaceMapping::TOTAL_CHIN_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::CHEEK_POINTS_START, FaceMapping::TOTAL_CHEEK_POINTS);
		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::BROW_POINTS_START, FaceMapping::TOTAL_BROW_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::NOSE_POINTS_START, FaceMapping::TOTAL_NOSE_POINTS);
		glDrawArrays(GL_TRIANGLE_FAN, FaceMapping::PHILTRUM_POINTS_START, FaceMapping::TOTAL_PHILTRUM_POINTS);
		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::FOREHEAD_POINTS_START, FaceMapping::TOTAL_FOREHEAD_POINTS);
		if (params.texturedMouth)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::MOUTH_POINTS_START, FaceMapping::TOTAL_MOUTH_POINTS);
		}
		if (params.texturedEyes)
		{
			glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::EYE_POINTS_START, FaceMapping::TOTAL_EYE_POINTS);
		}
	}

	if (params.lineAmbient[3] != 0 || params.lineTexId != 0 && params.lineDiffuse[3] != 0 && false)
	{
		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, lineVBO));
		VertexAttribSetter vTexCoord;

		bool textureExist = (false && params.lineTexId != 0);

		if (textureExist)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, params.lineTexId);

			vTexCoord = VertexAttribSetter(VertexAttribSetter::TexCoordAttribSetter(*shader, lineTexCoordVBO));
			shader->setUniform1i("hasTexture", 1);
		}
		else
		{
			shader->setUniform1i("hasTexture", 0);
		}

		glDrawArrays(GL_TRIANGLE_STRIP, 0, FaceMapping::LINE_STRIP_COUNT);
		glDrawArrays(GL_TRIANGLE_STRIP, FaceMapping::LINE_STRIP_COUNT, 4);
	}

	glPopAttrib();
}

void HalfFacePaint::unload()
{
	glDeleteBuffers(1, &leftVBO);
	glDeleteBuffers(1, &rightVBO);
	glDeleteBuffers(1, &lineVBO);

	glDeleteBuffers(1, &leftTexCoordVBO);
	glDeleteBuffers(1, &rightTexCoordVBO);
	glDeleteBuffers(1, &lineTexCoordVBO);

	glDeleteBuffers(1, &textureid);
	textureid = 0;
}

FxWidgetFacePaint::FxWidgetFacePaint()
{
	headTextureMap = false;
	zOffset.fill(0);
	match3D.fill(false);
}

void FxWidgetFacePaint::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (auto &face : halfFacePaint.facePointsSmoothers)
	{
		for (auto &smoother : face)
		{
			smoother.SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		}
	}
}

void FxWidgetFacePaint::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);
	bool txEyes = modelRecord.get<bool>("texturedEyes", false);
	bool txMouth = modelRecord.get<bool>("texturedMouth", false);
	for (auto &params : halfFaceParams)
	{
		params.texturedEyes = txEyes;
		params.texturedMouth = txMouth;
	}
	headTextureMap = modelRecord.get<bool>("headTextureMap", false);
}

void FxWidgetFacePaint::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
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

	match3D[targetIndex] = suit.get<bool>("match3D", false);
	zOffset[targetIndex] = suit.get<float>("zOffset", 0);

	halfFaceParams[targetIndex].applySuit(suit, loadTexturesImmediately);
}

bool FxWidgetFacePaint::load()
{
	FaceModel = FaceMapping::getFaceModel();

	for (auto &params : halfFaceParams)
	{
		params.load();
	}

	halfFacePaint.load(headTextureMap);

	return true;
}

void FxWidgetFacePaint::transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	Matrix4f projection = Matrix4f::Zero();
	Matrix3f anti_effectRotation;

	auto rollSmooth = renderParams.rollSmoother[face.pointId].smooth(face.roll);
	auto pitchSmooth = renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
	auto yawSmooth = renderParams.yawSmoother[face.pointId].smooth(face.yaw);

	projection(0, 0) = 2 / width;
	projection(1, 1) = -2 / height;
	projection(3, 3) = 1;
	projection(0, 3) = -1;
	projection(1, 3) = 1;

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

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double objWidth = widthRawSmooth / face.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix(Affine3d(Translation3d(Vector3d(0, 0, -distance + zOffset[face.pointId]))).matrix());

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = Matrix3f::Identity();
	renderParams.yawMatrix = Matrix3f::Identity();

	halfFacePaint.transform(face, halfFaceParams[face.pointId]);
}

void FxWidgetFacePaint::draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	ObjectRenderParams *objParams = objectRenderParams[face.pointId][0].get();
	auto shader = objParams->shader;

	ShaderSetter shaderSetter(shader, *this);

	SetUniformsForObject(*shader, 0, face.pointId);

	glActiveTexture(GL_TEXTURE10);
	glBindTexture(GL_TEXTURE_2D, cameraTextureId);

	int alphaMask = shader->GetUniformLocation("OriginFrame");
	glUniform1i(alphaMask, 10);

	shader->setUniform1i("frameWidth", Resolutions::INPUT_ACTUAL_WIDTH);
	shader->setUniform1i("frameHeight", Resolutions::INPUT_ACTUAL_HEIGHT);

	shader->setUniform1f("faceBrightness", externalRenderParams.faceParams.brightness);

	float frameBrightness = std::max(externalRenderParams.frameParams.brightness, externalRenderParams.faceParams.brightness * 1.5f);
	shader->setUniform1f("frameBrightness", frameBrightness);
	shader->setUniform3f("frameLight", externalRenderParams.frameParams.light[2], externalRenderParams.frameParams.light[1], externalRenderParams.frameParams.light[0]);

	halfFacePaint.draw(face, halfFaceParams[face.pointId], shader);
}

void FxWidgetFacePaint::unload()
{
	halfFacePaint.unload();
}

std::shared_ptr<ObjectRenderParams> FxWidgetFacePaint::createDefaultObjectRenderParams()
{
	return std::make_shared<ObjectRenderParams>();
}

void FxWidgetFacePaint::setDefaultObjectRenderParams(ObjectRenderParams& params)
{
}
