#include <fx/FxDragon.h>

FxDragon::FxDragon()
{
	auto shader3D = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/normalMapVertexShader.txt", "./assets/shaders/fragment/normalMapPhongFragmentShader.txt");

	auto headModel = make_shared<FxWidget3D>();
	headModel->modelPath = "./assets/fx/dragon/models/dragonTopHead_2.obj";
	headModel->texturesPaths.fill({ "./assets/fx/dragon/textures/Wyvern_Black.jpg" });
	headModel->name = "head";
	headModel->modelScale = 1.1f;

	auto jawsModel = make_shared<FxWidget3D>();
	jawsModel->modelPath = "./assets/fx/dragon/models/dragonBottomHead_2.obj";
	jawsModel->texturesPaths.fill( {"./assets/fx/dragon/textures/Wyvern_Black.jpg" });
	jawsModel->name = "jaws";
	jawsModel->modelScale = 1.1f;

	auto skinRenderParams = make_shared<ObjectRenderParams>();
	skinRenderParams->lightPos = { 0.2f, 1, 0.4f, 0 };
	skinRenderParams->cameraPos = { 0, 0, 0, 1 };
	skinRenderParams->ambientLight = { 0.4f, 0.4f, 0.4f };
	skinRenderParams->diffuseLight = { 0.4f, 0.4f, 0.4f };
	skinRenderParams->specularLight = { 1, 1, 1 };
	skinRenderParams->specularPower = 20;
	skinRenderParams->shader = shader3D;
	skinRenderParams->normalMap = "./assets/fx/dragon/textures/Wyvern_N.png";

	jawsModel->objectRenderParams.fill( { skinRenderParams} );
	headModel->objectRenderParams.fill({ skinRenderParams });

	models.push_back(headModel);
	models.push_back(jawsModel);
}

FxDragon::~FxDragon()
{
	
}

void FxDragon::onInputFrameResized()
{
	FX::onInputFrameResized();

	for (auto &smoother : lipsDistanceSmoother)
	{
		smoother.SMOOTH_MODIFIER = 1.0 / 60 * 1280.0 / Resolutions::INPUT_ACTUAL_WIDTH;
	}
}

void FxDragon::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	auto lipToLip = lipsDistanceSmoother[face.pointId].smooth(
		PointCalcUtil::distanceBetween(face.pts[61 * 2], face.pts[61 * 2 + 1], face.pts[64 * 2], face.pts[64 * 2 + 1]));

	const double STD_FACE_WIDTH = 0.172;
	const double STD_LIPS_DISTANCE = 60;

	auto factor = lipToLip / (face.width / STD_FACE_WIDTH * STD_LIPS_DISTANCE);

	double h = 30.0;

	double vAngle = externalRenderParams.vAngle;

	double f = 1 / tan(vAngle / 2);

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double hAngleHalf = atan(tan(vAngle / 2) * aspect);

	double planeH = 2.f *h * tan(vAngle / 2);
	double planeW = planeH * aspect;

	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;
	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(3, 2) = -1;
	projection(2, 3) = 2 * zFar*zNear / (zNear - zFar);

	for (int i = 0; i < 2; ++i)
	{
		models[i]->renderParams.projection = projection;

		auto xCenterRawSmooth = models[i]->renderParams.xCenterSmoother[face.pointId].smooth(face.xCenterRaw);
		auto yCenterRawSmooth = models[i]->renderParams.yCenterSmoother[face.pointId].smooth(face.yCenterRaw);

		auto rollSmooth = models[i]->renderParams.rollSmoother[face.pointId].smooth(face.roll);
		auto pitchSmooth = models[i]->renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
		auto yawSmooth = models[i]->renderParams.yawSmoother[face.pointId].smooth(face.yaw);

		auto widthRawSmooth = models[i]->renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

		double tx = xCenterRawSmooth / face.frameWidth;
		double ty = yCenterRawSmooth / face.frameHeight;

		double xShift = (tx - 0.5) * planeW;
		double yShift = -(ty - 0.5) * planeH;

		Matrix4d modelMatrix;

		modelMatrix = Affine3d(Translation3d(Vector3d(xShift, yShift, 0))).matrix();

		Vector3d rotateVector = Vector3d(0.0, 0.0, -h).cross(Vector3d(xShift, yShift, -h));
		rotateVector.normalize();
		double rotateAngle = atan((sqrt(xShift*xShift + yShift*yShift)) / h);

		double additionalRotation = factor * M_PI_4 * (i == 0 ? -1 : 1);

		Affine3d rotation (AngleAxisd(additionalRotation, Vector3d(1, 0, 0)));

		rotation *= AngleAxisd(rotateAngle, rotateVector);
		rotation *= AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
		rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
		rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

		modelMatrix *= rotation.matrix();

		const double widthToDepth = 1.0;
		double width = planeW * widthRawSmooth / face.frameWidth;
		double faceDepth = widthToDepth * width;
		modelMatrix *= Affine3d(Translation3d(0, 0, -faceDepth)).matrix();

		const double scaleK = 0.08;
		double scale = width * scaleK * ((FxWidget3D*)models[i].get())->modelScale;
		modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

		modelMatrix = Affine3d(Translation3d(0, 0, -h)).matrix() * modelMatrix;

		models[i]->depth = -faceDepth * scale - h;

		models[i]->renderParams.modelView = modelMatrix.cast<float>();
		models[i]->renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
		models[i]->renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
	}
}