#include <Widgets/FxWidgetParticleEmitter.h>
#include <QDebug>

const std::string FxWidgetParticleEmitter::TYPE_NAME = "ParticleEmitter";

std::string FxWidgetParticleEmitter::getTypeName()
{
	return TYPE_NAME;
}

void FxWidgetParticleEmitter::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		pivotX[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
		pivotY[i].SMOOTH_MODIFIER = 1.0 / ValueSmoother::SMOOTH_MODIFIER_XY * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
	}
}

FxWidgetParticleEmitter::FxWidgetParticleEmitter()
{
	bmin = { 0, 0, 0 };
	bmax = { 0, 0, 0 };

	backClipping = false;

	pivotOffset = { 0, 0, 0 };
}

boost::property_tree::ptree FxWidgetParticleEmitter::getPTree(ExtraModelData &data)
{
	auto tree = GraphicsModel::getPTree(data);

	tree.put("type", getTypeName());

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

void FxWidgetParticleEmitter::loadFromJSON(boost::property_tree::ptree& modelRecord)
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
}

void FxWidgetParticleEmitter::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
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
}

bool FxWidgetParticleEmitter::load()
{
	GraphicsModel::load();

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

	return true;
}

void FxWidgetParticleEmitter::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
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
	Matrix3d rotation(AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
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

void FxWidgetParticleEmitter::draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	for (auto &particle : particles[target.pointId])
	{
		Matrix4f projection = renderParams.projection;
		particle.effect.draw(projection);
	}

	glPopAttrib();
}

void FxWidgetParticleEmitter::unload()
{
	GraphicsModel::unload();
}
