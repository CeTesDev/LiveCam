#include <Widgets/FxModAlphaMask.h>

const std::string AlphaMask::TYPE_NAME = "AlphaMask";

std::string AlphaMask::getTypeName()
{
	return TYPE_NAME;
}

AlphaMask::AlphaMask()
{
	smoothEdgeWidthFactor.fill(5);
	cutOutLeftEye.fill(false);
	cutOutRightEye.fill(false);
	cutOutMouth.fill(false);
	cutOutChin.fill(false);
	outerLips.fill(false);

	shader2D = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/default.vertex", "./Assets/shaders/fragment/alphamask.frag");

	name = "AlphaMask";
}

AlphaMask::~AlphaMask() { }

void AlphaMask::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		renderParams.xCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		renderParams.yCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		renderParams.widthSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY / 2 * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;

		renderParams.pitchSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		renderParams.yawSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		renderParams.rollSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
	}

	for (auto &face : facePointsSmoothers)
	{
		for (auto &smoother : face)
		{
			smoother.SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		}
	}
}

void AlphaMask::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);

	canSwapSuit = modelRecord.get<bool>("canSwapSuit", true);

	shader2D = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/default.vertex", "./Assets/shaders/fragment/alphamask.frag");

	name = "AlphaMask";
}

void AlphaMask::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	smoothEdgeWidthFactor[targetIndex] = suit.get<float>("smoothEdgeWidth", 5);
	cutOutLeftEye[targetIndex] = suit.get<bool>("cutOutLeftEye", false);
	cutOutRightEye[targetIndex] = suit.get<bool>("cutOutRightEye", false);
	cutOutMouth[targetIndex] = suit.get<bool>("cutOutMouth", false);
	cutOutChin[targetIndex] = suit.get<bool>("cutOutChin", false);
	outerLips[targetIndex] = suit.get<bool>("outerLips", false);
}

const std::array<int, AlphaMask::FAN_LIMIT> AlphaMask::fan_modifiers =
{
	8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 19, 18, 17, 0, 1, 2, 3, 4, 5, 6, 7
};

const std::array<int, AlphaMask::LEFT_EYE_STRIP_COUNT> AlphaMask::left_eye_strip =
{
	36, 37, 41, 38, 40, 39
};

const std::array<int, AlphaMask::RIGHT_EYE_STRIP_COUNT> AlphaMask::right_eye_strip =
{
	45, 46, 44, 47, 43, 42
};

const std::array<int, AlphaMask::LIPS_INNER_STRIP_COUNT> AlphaMask::lips_inner_strip =
{
	60, 65, 61, 64, 62, 63
};

const std::array<int, AlphaMask::LIPS_OUTER_STRIP_COUNT> AlphaMask::lips_outer_strip =
{
	48, 49, 59, 50, 58, 51, 57, 52, 56, 53, 55, 54
};

const std::array<int, AlphaMask::CHIN_FAN_COUNT> AlphaMask::chin_fan =
{
	8, 9, 10, 54, 48, 6, 7
};

bool AlphaMask::load()
{
	glGenBuffers(1, &faceMask_VBO);

	return true;
}

void AlphaMask::transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double objWidth = widthRawSmooth / face.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = 2 / width;
	projection(1, 1) = -2 / height;
	projection(3, 3) = 1;
	projection(0, 3) = -1;
	projection(1, 3) = 1;
	projection(2, 2) = 2 / (zNear - zFar);
	projection(2, 3) = (zFar + zNear) / (zNear - zFar);

	renderParams.projection = projection;

	Matrix4d modelMatrix(Affine3d(Translation3d(Vector3d(0, 0, depth))).matrix());

	renderParams.modelView = modelMatrix.cast<float>();
	renderParams.rotationMatrix = Matrix3f::Identity();
	renderParams.yawMatrix = Matrix3f::Identity();

	std::array<float, TARGET_DETAIL_LIMIT * 2> smoothedPoints;

	for (int i = 0; i < TARGET_DETAIL_LIMIT; ++i)
	{
		smoothedPoints[i * 2] = facePointsSmoothers[face.pointId][i * 2].smooth(face.pts[i * 2]);
		smoothedPoints[i * 2 + 1] = facePointsSmoothers[face.pointId][i * 2 + 1].smooth(face.pts[i * 2 + 1]);
	}

	ImagePoints points(smoothedPoints);

	cv::Point2f chinPoint = points.at(9);
	cv::Point2f tmp = PointCalcUtil::crossPointTwoLines(chinPoint, points.at(28), points.at(21), points.at(24));
	Eigen::Vector2f topPoint = { tmp.x, tmp.y };

	Eigen::Vector2f verticalOffset = points.getEigenAt(28) - Eigen::Vector2f(chinPoint.x, chinPoint.y);
	verticalOffset *= 0.2;

	const int total = FAN_LIMIT + LEFT_EYE_STRIP_COUNT + RIGHT_EYE_STRIP_COUNT + LIPS_INNER_STRIP_COUNT + LIPS_OUTER_STRIP_COUNT +
		CHIN_FAN_COUNT;
	std::array<float, total * 3> buffer;

	auto iter = buffer.begin();

	auto iterFace = face_contour.begin();
	for (auto index : fan_modifiers)
	{
		if (index < 17)
		{
			Eigen::Vector2f v = { points.pts[index * 2], points.pts[index * 2 + 1] };
			*iter++ = v[0];
			*iter++ = v[1];
			*iterFace++ = v;
		}
		else
		{
			Eigen::Vector2f v = { points.pts[index * 2] + verticalOffset[0], points.pts[index * 2 + 1] + verticalOffset[1] };
			*iter++ = v[0];
			*iter++ = v[1];
			*iterFace++ = v;
		}
		*iter++ = 0;
	}
	*iterFace = { points.pts[8 * 2], points.pts[8 * 2 + 1] };

	if (cutOutLeftEye[face.pointId])
	{
		auto leftEyeCenter = PointCalcUtil::crossPointTwoLines(points.at(38), points.at(41), points.at(39), points.at(42));

		for (auto index : left_eye_strip)
		{
			auto newPoint = leftEyeCenter + (points.at(index + 1) - leftEyeCenter) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
			*iter++ = newPoint.x;
			*iter++ = newPoint.y;
			*iter++ = 0;
		}

		left_eye_contour = { points.getEigenAt(37), points.getEigenAt(42), points.getEigenAt(41),
			points.getEigenAt(40), points.getEigenAt(39), points.getEigenAt(38), points.getEigenAt(37) };

		for (auto &point : left_eye_contour)
		{
			Eigen::Vector2f eigenVec = { leftEyeCenter.x, leftEyeCenter.y };
			point = eigenVec + (point - eigenVec) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
		}
	}
	else
	{
		iter += 3 * LEFT_EYE_STRIP_COUNT;
	}

	if (cutOutRightEye[face.pointId])
	{
		auto rightEyeCenter = PointCalcUtil::crossPointTwoLines(points.at(44), points.at(47), points.at(45), points.at(48));
		for (auto index : right_eye_strip)
		{
			auto newPoint = rightEyeCenter + (points.at(index + 1) - rightEyeCenter) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
			*iter++ = newPoint.x;
			*iter++ = newPoint.y;
			*iter++ = 0;
		}

		right_eye_contour = { points.getEigenAt(43), points.getEigenAt(48), points.getEigenAt(47),
			points.getEigenAt(46), points.getEigenAt(45), points.getEigenAt(44), points.getEigenAt(43) };

		for (auto &point : right_eye_contour)
		{
			Eigen::Vector2f eigenVec = { rightEyeCenter.x, rightEyeCenter.y };
			point = eigenVec + (point - eigenVec) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
		}
	}
	else
	{
		iter += 3 * RIGHT_EYE_STRIP_COUNT;
	}

	if (cutOutChin[face.pointId])
	{
		auto chinCenter = PointCalcUtil::crossPointTwoLines(points.at(49), points.at(11), points.at(7), points.at(55));
		for (auto index : chin_fan)
		{
			auto newPoint = chinCenter + (points.at(index + 1) - chinCenter) * (1 + 0.3 * smoothEdgeWidthFactor[face.pointId]);
			*iter++ = newPoint.x;
			*iter++ = newPoint.y;
			*iter++ = 0;
		}

		chin_contour = { points.getEigenAt(9), points.getEigenAt(10), points.getEigenAt(11),
			points.getEigenAt(55), points.getEigenAt(49), points.getEigenAt(7), points.getEigenAt(8) };

		for (auto &point : chin_contour)
		{
			Eigen::Vector2f eigenVec = { chinCenter.x, chinCenter.y };
			point = eigenVec + (point - eigenVec) * (1 + 0.3 * smoothEdgeWidthFactor[face.pointId]);
		}
	}
	else
	{
		iter += 3 * CHIN_FAN_COUNT;
	}

	if (cutOutMouth[face.pointId])
	{
		cv::Point2f mouthCenter = PointCalcUtil::centerOf2Points(points.at(51), points.at(57));
		if (outerLips[face.pointId])
		{
			iter += 3 * LIPS_INNER_STRIP_COUNT;

			for (auto index : lips_outer_strip)
			{
				auto newPoint = mouthCenter + (points.at(index + 1) - mouthCenter) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
				*iter++ = newPoint.x;
				*iter++ = newPoint.y;
				*iter++ = 0;
			}
			lips_outer_contour = { points.getEigenAt(49), points.getEigenAt(60), points.getEigenAt(59),
				points.getEigenAt(58), points.getEigenAt(57), points.getEigenAt(56),
				points.getEigenAt(55),points.getEigenAt(54), points.getEigenAt(53),
				points.getEigenAt(52), points.getEigenAt(51), points.getEigenAt(50), points.getEigenAt(49) };

			for (auto &point : lips_outer_contour)
			{
				Eigen::Vector2f eigenVec = { mouthCenter.x, mouthCenter.y };
				point = eigenVec + (point - eigenVec) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
			}
		}
		else
		{
			for (auto index : lips_inner_strip)
			{
				auto newPoint = mouthCenter + (points.at(index + 1) - mouthCenter) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
				*iter++ = newPoint.x;
				*iter++ = newPoint.y;
				*iter++ = 0;
			}
			lips_inner_contour = { points.getEigenAt(61), points.getEigenAt(66), points.getEigenAt(65),
				points.getEigenAt(64), points.getEigenAt(63), points.getEigenAt(62), points.getEigenAt(61) };

			for (auto &point : lips_inner_contour)
			{
				Eigen::Vector2f eigenVec = { mouthCenter.x, mouthCenter.y };
				point = eigenVec + (point - eigenVec) * (1 + 0.4 * smoothEdgeWidthFactor[face.pointId]);
			}
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, faceMask_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * total * 3, buffer.data(), GL_STREAM_DRAW);
}

void AlphaMask::draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	auto prevColorBuffer = frameManager.getCurrentDrawBuffer();
	frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::ALPHA_MASK);

	glDisable(GL_ALPHA_TEST);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFuncSeparate(GL_ZERO, GL_ONE, GL_ONE, GL_ZERO);

	ShaderSetter shaderSetter(shader2D, *this);

	VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader2D, faceMask_VBO));

	float edgeFadingDistance = face.width * 0.05f * face.frameWidth * smoothEdgeWidthFactor[face.pointId];
	shader2D->setUniform1f("edgeFadingDistance", edgeFadingDistance);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, frameManager.getFrameRenderBuffer().alphaMaskId);

	auto AlphaMaskLoc = shader2D->GetUniformLocation("AlphaMask");
	glUniform1i(AlphaMaskLoc, 1);

	shader2D->setUniform1i("WinWidth", Resolutions::OUTPUT_WIDTH);
	shader2D->setUniform1i("WinHeight", Resolutions::OUTPUT_HEIGHT);

	shader2D->setUniform1i("edgesCount", 24);
	shader2D->setUniform1f("newAlpha", 0);
	PointCalcUtil::LoadLinesToUniform(face_contour, "edges", shader2D);
	glDrawArrays(GL_TRIANGLE_FAN, 0, FAN_LIMIT);

	glFinish();

	if (cutOutLeftEye[face.pointId])
	{
		shader2D->setUniform1i("edgesCount", 6);
		shader2D->setUniform1f("newAlpha", 1);
		PointCalcUtil::LoadLinesToUniform(left_eye_contour, "edges", shader2D);
		glDrawArrays(GL_TRIANGLE_STRIP, FAN_LIMIT, LEFT_EYE_STRIP_COUNT);
	}

	glFinish();

	if (cutOutRightEye[face.pointId])
	{
		shader2D->setUniform1i("edgesCount", 6);
		shader2D->setUniform1f("newAlpha", 1);
		shader2D->setUniform1f("oldAlpha", 0);
		PointCalcUtil::LoadLinesToUniform(right_eye_contour, "edges", shader2D);
		glDrawArrays(GL_TRIANGLE_STRIP, FAN_LIMIT + LEFT_EYE_STRIP_COUNT, RIGHT_EYE_STRIP_COUNT);
	}

	glFinish();

	if (cutOutChin[face.pointId])
	{
		shader2D->setUniform1i("edgesCount", 7);
		shader2D->setUniform1f("newAlpha", 1);
		shader2D->setUniform1f("oldAlpha", 0);
		PointCalcUtil::LoadLinesToUniform(chin_contour, "edges", shader2D);
		glDrawArrays(GL_TRIANGLE_FAN, FAN_LIMIT + LEFT_EYE_STRIP_COUNT + RIGHT_EYE_STRIP_COUNT, CHIN_FAN_COUNT);
	}

	glFinish();

	if (cutOutMouth[face.pointId])
	{
		if (outerLips[face.pointId])
		{
			shader2D->setUniform1i("edgesCount", 12);
			shader2D->setUniform1f("newAlpha", 1);
			PointCalcUtil::LoadLinesToUniform(lips_outer_contour, "edges", shader2D);
			glDrawArrays(GL_TRIANGLE_STRIP, LIPS_INNER_STRIP_COUNT + CHIN_FAN_COUNT + FAN_LIMIT + LEFT_EYE_STRIP_COUNT + RIGHT_EYE_STRIP_COUNT, LIPS_OUTER_STRIP_COUNT);
		}
		else
		{
			shader2D->setUniform1i("edgesCount", 6);
			shader2D->setUniform1f("newAlpha", 1);
			PointCalcUtil::LoadLinesToUniform(lips_inner_contour, "edges", shader2D);
			glDrawArrays(GL_TRIANGLE_STRIP, CHIN_FAN_COUNT + FAN_LIMIT + LEFT_EYE_STRIP_COUNT + RIGHT_EYE_STRIP_COUNT, LIPS_INNER_STRIP_COUNT);
		}
	}

	frameManager.SwitchToDrawBuffer(prevColorBuffer);

	glPopAttrib();
}

void AlphaMask::unload()
{
	GraphicsModel::unload();

	glDeleteBuffers(1, &faceMask_VBO);
}