#include <Widgets/FxModDepthMask.h>

const std::string DepthMask::TYPE_NAME = "DepthMask";

std::string DepthMask::getTypeName()
{
	return TYPE_NAME;
}

DepthMask::DepthMask() : DepthMask(0, true, true)
{
}

DepthMask::DepthMask(float zOffset, bool draw2Dmask, bool draw3Dmask)
{
	this->zOffset = zOffset;
	this->draw2Dmask = draw2Dmask;
	this->draw3Dmask = draw3Dmask;

	shader2D = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/depthmask.vertex", "./assets/shaders/fragment/transparent.frag");

	shader3D = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/transparent.vertex", "./assets/shaders/fragment/transparent.frag");

	name = "depthMask";

	canSwapSuit = false;

	modelPath = "./assets/fx/depthMask/head_no_ears.obj";

	modelScale = 1.9f;
	modelShift = { 0, -0.33f, 0 };
}

DepthMask::~DepthMask() { }

void DepthMask::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		params2D.xCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params2D.yCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params2D.widthSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY / 2 * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params2D.pitchSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		params2D.yawSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		params2D.rollSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;

		params3D.xCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params3D.yCenterSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params3D.widthSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY / 2 * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		params3D.pitchSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		params3D.yawSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
		params3D.rollSmoother[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_Z;
	}
}

void DepthMask::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	canSwapSuit = modelRecord.get<bool>("canSwapSuit", false);

	this->zOffset = modelRecord.get<float>("zOffset", 0);
	this->draw2Dmask = modelRecord.get<bool>("draw2DMask", true);
	this->draw3Dmask = modelRecord.get<bool>("draw3DMask", true);

	shader2D = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/depthmask.vertex", "./assets/shaders/fragment/transparent.frag");

	shader3D = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/transparent.vertex", "./assets/shaders/fragment/transparent.frag");

	name = "depthMask";

	bool toStealEars = !modelRecord.get<bool>("drawEars", false);

	modelPath = modelRecord.get<std::string>("OBJ",
		toStealEars ? "./assets/fx/depthMask/head_no_ears.obj" : "./assets/fx/depthMask/head.obj");

	auto extraShiftVector = modelRecord.get_child_optional("modelShift");
	modelShift = extraShiftVector ? JSONVectorReader::readVector3f(extraShiftVector.get()) : Eigen::Vector3f(0, -0.33f, 0);

	modelScale = modelRecord.get<float>("modelScale", 1.9f);
}

void DepthMask::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	auto extraShiftVector = suit.get_child_optional("modelShift");
	if (extraShiftVector)
	{
		modelShift = JSONVectorReader::readVector3f(extraShiftVector.get());
	}

	auto modelScaleRecord = suit.get_optional<float>("modelScale");
	if (modelScaleRecord)
	{
		modelScale = modelScaleRecord.get();
	}

	auto zOffset = suit.get_optional<float>("zOffset");
	if (zOffset)
	{
		this->zOffset = zOffset.get();
	}

	auto draw2Dmask = suit.get_optional<bool>("draw2DMask");
	if (draw2Dmask)
	{
		this->draw2Dmask = draw2Dmask.get();
	}

	auto draw3Dmask = suit.get_optional<bool>("draw3DMask");
	if (draw3Dmask)
	{
		this->draw3Dmask = draw3Dmask.get();
	}
}

const std::array<int, DepthMask::FAN_LIMIT> DepthMask::fan_modifiers =
{
	8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 19, 18, 17, 0, 1, 2, 3, 4, 5, 6, 7
};

bool DepthMask::load()
{
	
	FxWidget3D::load();

	glGenBuffers(1, &mask2dVBO);

	return true;
}

void DepthMask::transform(TrackingTarget& targetModel, ExternalRenderParams &externalRenderParams)
{
	if (!draw2Dmask && !draw3Dmask) return;

	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	auto widthRawSmooth = params3D.widthSmoother[targetModel.pointId].smooth(targetModel.widthRaw);

	double objWidth = widthRawSmooth / targetModel.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	if (draw3Dmask)
	{
		double vAngle = externalRenderParams.vAngle;
		double f = 1 / tan(vAngle / 2);
		double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

		Matrix4f projection = Matrix4f::Zero();

		projection(0, 0) = f / aspect;
		projection(1, 1) = f;
		projection(2, 2) = (zFar + zNear) / (zNear - zFar);
		projection(3, 2) = -1;
		projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

		params3D.projection = projection;

		auto xCenterRawSmooth = renderParams.xCenterSmoother[targetModel.pointId].smooth(targetModel.xCenterRaw);
		auto yCenterRawSmooth = renderParams.yCenterSmoother[targetModel.pointId].smooth(targetModel.yCenterRaw);

		auto rollSmooth = renderParams.rollSmoother[targetModel.pointId].smooth(targetModel.roll);
		auto pitchSmooth = renderParams.pitchSmoother[targetModel.pointId].smooth(targetModel.pitch);
		auto yawSmooth = renderParams.yawSmoother[targetModel.pointId].smooth(targetModel.yaw);

		double tx = xCenterRawSmooth / targetModel.frameWidth;
		double ty = yCenterRawSmooth / targetModel.frameHeight;

		double hAngleHalf = atan(tan(vAngle / 2) * aspect);

		Matrix4d modelMatrix;

		double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
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

		Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
		rotation *= AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
		rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
		rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

		modelMatrix *= rotation.matrix();

		Eigen::Matrix4d headAdjusting = Eigen::Matrix4d::Identity();

		shift = { modelShift[0] * STD_FACE_WIDTH,  modelShift[1] * STD_FACE_WIDTH, modelShift[2] * STD_HEAD_LENGTH };
		headAdjusting *= Affine3d(Translation3d(Vector3d(shift[0], shift[1], shift[2] - STD_HEAD_LENGTH / 2))).matrix();

		double scale = objWidth / (bmax[0] - bmin[0]) * distance / STD_FACE_DISTANCE * modelScale;
		headAdjusting *= Affine3d(Scaling(scale, scale, scale)).matrix();

		Vector3d autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
		headAdjusting *= Affine3d(Translation3d(autoShift)).matrix();

		modelMatrix *= headAdjusting;

		params3D.additionalMatrices4[0] = headAdjusting.cast<float>();

		params3D.modelView = modelMatrix.cast<float>();
		params3D.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
		params3D.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
	}

	if (draw2Dmask)
	{

		Matrix4f projection = Matrix4f::Zero();
		projection(0, 0) = 2 / width;
		projection(1, 1) = -2 / height;
		projection(2, 2) = (zFar + zNear) / (zNear - zFar);
		projection(3, 3) = 1;
		projection(0, 3) = -1;
		projection(1, 3) = 1;
		projection(2, 3) = 2 * zFar * zNear / (zNear - zFar);

		params2D.projection = projection;

		Matrix4d modelMatrix(Affine3d(Translation3d(Vector3d(0, 0, depth + zOffset))).matrix());

		params2D.modelView = modelMatrix.cast<float>();
		params2D.rotationMatrix = Matrix3f::Identity();
		params2D.yawMatrix = Matrix3f::Identity();

		ImagePoints points(targetModel.pts);

		cv::Point2f chinPoint = points.at(9);
		cv::Point2f tmp = PointCalcUtil::crossPointTwoLines(chinPoint, points.at(28), points.at(21), points.at(24));
		Eigen::Vector2f topPoint = { tmp.x, tmp.y };

		Eigen::Vector2f verticalOffset = points.getEigenAt(28) - Eigen::Vector2f(chinPoint.x, chinPoint.y);
		verticalOffset *= 0.2;

		std::array<float, FAN_LIMIT * 3> buffer;
		auto iter = buffer.begin();

		for (auto index : fan_modifiers)
		{
			if (index < 17)
			{
				*iter++ = points.pts[index * 2];
				*iter++ = points.pts[index * 2 + 1];
			}
			else
			{
				*iter++ = points.pts[index * 2] + verticalOffset[0];
				*iter++ = points.pts[index * 2 + 1] + verticalOffset[1];
			}
			*iter++ = 0;
		}

		glBindBuffer(GL_ARRAY_BUFFER, mask2dVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * FAN_LIMIT * 3, buffer.data(), GL_STREAM_DRAW);
	}
}

void DepthMask::draw(TrackingTarget& targetModel, ExternalRenderParams &externalRenderParams)
{
	if (!draw2Dmask && !draw3Dmask) return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);

	if (draw2Dmask)
	{
		shader2D->begin();

		params2D.SetUniforms(*shader2D);

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader2D, mask2dVBO));

		glDrawArrays(GL_TRIANGLE_FAN, 0, FAN_LIMIT);

		shader2D->end();
	}

	if (draw3Dmask)
	{
		shader3D->begin();

		params3D.SetUniforms(*shader3D);

		GraphicsData o = objects[0];

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader3D, o.vb));

		glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);

		shader3D->end();
	}

	glPopAttrib();
}

void DepthMask::unload()
{
	FxWidget3D::unload();

	glDeleteBuffers(1, &mask2dVBO);
}