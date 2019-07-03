#include <fx/FxBravesTomahawkChop.h>

FxBravesTomahawkChop::FxBravesTomahawkChop()
{
	animationCounter.fill(0);
	animationDirection.fill(false);

	auto shader = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/simpleVertexShader.txt", "./assets/shaders/fragment/phongFragmentShader.txt");

	auto headModel = make_shared<FxWidget3D>();
	headModel->modelPath = "./assets/fx/atlanta_braves/animated_tomahawk/BravesTomaHawk_V9.obj";
	headModel->texturesPaths.fill({ "./assets/fx/atlanta_braves/animated_tomahawk/balance_cube.png", "./assets/fx/atlanta_braves/animated_tomahawk/BravesLogoUV2.png" });
	headModel->name = "animated_tomahawk";
	headModel->modelScale = 1.0f;
	headModel->modelShift = { 0, 0.3f, 0.15f };

	auto hatRenderParams = make_shared<ObjectRenderParams>();
	hatRenderParams->lightPos = { 0.2f, 0.4f, 1.0f, 0 };
	hatRenderParams->cameraPos = { 0, 0, 0, 1 };
	hatRenderParams->ambientLight = { 0.6f, 0.6f, 0.6f };
	hatRenderParams->diffuseLight = { 0.6f, 0.6f, 0.6f };
	hatRenderParams->specularLight = { 0.15f, 0.15f, 0.15f };
	hatRenderParams->cullFace = false;
	hatRenderParams->alphaTest = true;
	hatRenderParams->specularPower = 5;
	hatRenderParams->shader = shader;
	
	headModel->objectRenderParams.fill({ hatRenderParams });

	models.push_back(headModel);
}

FxBravesTomahawkChop::~FxBravesTomahawkChop()
{

}

void FxBravesTomahawkChop::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	animateNextStep(face.pointId);

	auto lipToLip = lipsDistanceSmoother[face.pointId].smooth(
		PointCalcUtil::distanceBetween(face.pts[61 * 2], face.pts[61 * 2 + 1], face.pts[64 * 2], face.pts[64 * 2 + 1]));

	const double STD_FACE_WIDTH = 0.172;
	const double STD_LIPS_DISTANCE = 60;

	auto factor = 0.1 * (animationCounter[face.pointId]);

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

	for (int i = 0; i < 1; ++i)
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

		double xShift = ((tx - 0.5) * planeW);
		double yShift = (0 - (ty - 0.5) * planeH);

		Matrix4d modelMatrix;

		modelMatrix = Affine3d(Translation3d(Vector3d(xShift, yShift, 0))).matrix();

		Vector3d rotateVector = Vector3d(0.0, 0.0, -h).cross(Vector3d(xShift, yShift, -h));
		rotateVector.normalize();
		double rotateAngle = atan((sqrt(xShift*xShift + yShift*yShift)) / h);
		double additionalRotation = factor * M_PI_4;

		Affine3d rotation(AngleAxisd(additionalRotation, Vector3d(0, 0, 1)));

		rotation *= AngleAxisd(rotateAngle, rotateVector);

		modelMatrix *= rotation.matrix();

		const double widthToDepth = 1.0;
		double width = planeW * widthRawSmooth / face.frameWidth;
		double faceDepth = widthToDepth * width;
		modelMatrix *= Affine3d(Translation3d(0, 0, -faceDepth)).matrix();

		const double scaleK = 0.012;
		double scale = width * scaleK * ((FxWidget3D*)models[i].get())->modelScale;
		modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();

		modelMatrix = Affine3d(Translation3d(0, 0, -h)).matrix() * modelMatrix;

		models[i]->depth = -faceDepth * scale - h;

		models[i]->renderParams.modelView = modelMatrix.cast<float>();
		models[i]->renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	}
}
void FxBravesTomahawkChop::animateNextStep(size_t i)
{
	if (animationDirection[i]) {

		animationCounter[i]++;

		if (animationCounter[i] == 0)
		{
			animationDirection[i] = false;
		}
	}
	else {

		animationCounter[i]--;

		if (animationCounter[i] == (-20))
		{
			animationDirection[i] = true;
		}
	}
}