#include <fx/FxTeslaHelmetOKC.h>

#include <stdlib.h>

FxTeslaHelmetOKC::FxTeslaHelmetOKC()
{
	loadFromJSON("./assets/fx/okc_thunder/okc_tesla_helmet/okc_tesla_helmet_modules.json");

	auto helmetShader = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/backClippingNotCenteredVertexShader.txt", "./assets/shaders/fragment/backClippingPhongFragmentShader.txt");

	helmet = make_shared<FxWidget3D>();
	
	helmet->modelPath = "./assets/fx/okc_thunder/okc_tesla_helmet/Helmet_setup_with_lightning_2.obj";
	helmet->modelShift = { 0, 1.27f, -0.23f };
	helmet->modelScale = 2.02;
	
	helmet->texturesPaths.fill( { "./assets/fx/okc_thunder/okc_tesla_helmet/tex_2/phong1_Base_Color.png" });
	helmet->name = "helmet";

	helmetRenderParams = make_shared<ObjectRenderParams>();
	helmetRenderParams->cameraPos = { 0, 0, 0, 1 };
	helmetRenderParams->ambientLight = { 1, 1, 1 };
	helmetRenderParams->specularPower = 7;
	helmetRenderParams->blend = true;
	helmetRenderParams->shader = helmetShader;

	auto &addUniforms = helmetRenderParams->additionalUniforms;
	addUniforms["ZClip"] = TUniform1f(0);
	addUniforms["YClip"] = TUniform1f(0);
	addUniforms["ZRotation"] = TUniform3f(Eigen::Vector3f(0, 0, 0));

	helmet->objectRenderParams.fill( { helmetRenderParams });

	models.push_back(helmet);

	boltSprite.animationPath = "./assets/fx/okc_thunder/okc_tesla_helmet/lighting_sequence/lighting2/";
	boltSprite.animationIndex = 0;
	boltSprite.shader = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/spriteVertexShader.txt", "./assets/shaders/fragment/spriteFragmentShader.txt");
	
	srand(time(NULL));
	for (int i = 0; i < ALPHA_COUNT; ++i)
	{
		for (int p = 0; p < PHI_COUNT; ++p)
		{
			animationStart[i][p] = rand() % MAX_RELOADING_TIME + 1;
			animationIndex[i][p] = -1;
		}
	}
}

FxTeslaHelmetOKC::~FxTeslaHelmetOKC()
{

}

void FxTeslaHelmetOKC::load()
{
	FX::load();

	for (int i = 0; i < ObjectTracker::MAX_TO_TRACK; ++i)
	{
		auto &addUniforms = helmet->objectRenderParams[i][0]->additionalUniforms;
		addUniforms["XClip"] = TUniform1f(std::numeric_limits<float>::max());
		addUniforms["ZClip"] = TUniform1f(helmet->bmin[2] + (helmet->bmax[2] - helmet->bmin[2]) * 0.5f);
		addUniforms["YClip"] = TUniform1f(helmet->bmin[1] + (helmet->bmax[1] - helmet->bmin[1]) * 0.1f);
	}

	float scale = (helmet->bmax[0] - helmet->bmin[0]) * 1.5f;
	boltSprite.animationScale = { scale, scale };
	boltSprite.animationOffset = { helmet->bmin[0], helmet->bmax[1], 0 };

	boltSprite.load();
}

void FxTeslaHelmetOKC::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	FX::transform(face, externalRenderParams);
	

	Eigen::Vector4f minimum = { helmet->bmin[0], helmet->bmin[1], helmet->bmin[2], 1 };
	Eigen::Vector4f maximum = { helmet->bmax[0], helmet->bmax[1], helmet->bmax[2], 1 };

	Eigen::Matrix3f inversedRotation = helmet->renderParams.rotationMatrix.inverse();

	auto xRotationData = inversedRotation.block<1, 3>(0, 0).data();
	Eigen::Vector3f xRotation = { xRotationData[0], xRotationData[1], xRotationData[2] };

	auto zRotationData = helmet->renderParams.rotationMatrix.block<1, 3>(2, 0).data();
	Eigen::Vector3f zRotation = { zRotationData[0], zRotationData[1], zRotationData[2] };

	Eigen::Matrix4f headAdjusting = helmet->renderParams.additionalMatrices4[0];

	minimum = headAdjusting * minimum;
	maximum = headAdjusting * maximum;

	float Xclip = (maximum[0] - minimum[0]) / 2 * std::numeric_limits<float>::max();
	float Yclip = minimum[1] + (maximum[1] - minimum[1]) * 0.1f;
	float Zclip = minimum[2] + (maximum[2] - minimum[2]) * 0.5f;

	auto &param = helmet->objectRenderParams[face.pointId];

	param[0]->additionalUniforms["XClip"] = TUniform1f(Xclip);
	param[0]->additionalUniforms["YClip"] = TUniform1f(Yclip);
	param[0]->additionalUniforms["ZClip"] = TUniform1f(Zclip);
	param[0]->additionalUniforms["XRotation"] = TUniform3f(xRotation);
	param[0]->additionalUniforms["ZRotation"] = TUniform3f(zRotation);
	param[0]->additionalUniforms["HeadAdjustingMatrix"] = TUniform16(headAdjusting);

	float factor = 0;
	
	for (int i = 0; i < ALPHA_COUNT; ++i)
	{
		for (int p = 0; p < PHI_COUNT; ++p)
		{
			factor += animationIndex[i][p] == -1 ? 0 : 3.f * animationIndex[i][p] / (boltSprite.animationLength - 1.f) / PHI_COUNT / ALPHA_COUNT;

			if (animationStart[i][p] > 0)
			{
				--animationStart[i][p];
			}
			else
			{
				animationIndex[i][p] += 1;

				if (animationIndex[i][p] >= boltSprite.animationLength)
				{
					animationIndex[i][p] = -1;
					animationStart[i][p] = rand() % MAX_RELOADING_TIME + 1;
				}
			}
		}
	}

 	helmetRenderParams->lightPos = helmet->renderParams.modelView * Vector4f(0, 19.82f, 0, 0);
 	helmetRenderParams->diffuseLight = { 0.0706f * factor, 0.8589f * factor, 0.9725f * factor };
 	helmetRenderParams->specularLight = { 0.0353f * factor, 0.4295f * factor, 0.4862f * factor };
}

void FxTeslaHelmetOKC::draw(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	FX::draw(face, externalRenderParams);

	glDisable(GL_DEPTH_TEST);
	

	double phi_delta = M_PI_2 / (PHI_COUNT + 1);
	double phi = -M_PI_4 + phi_delta;

	for (int p = 0; p < PHI_COUNT; ++p, phi += phi_delta)
	{
		Matrix4d rotationZ;
		rotationZ = Affine3d(AngleAxisd(phi, Vector3d(0, 0, 1))).matrix();

		double alpha_delta = M_PI / ALPHA_COUNT;
		double alpha = 0;

		for (int i = 0; i < ALPHA_COUNT; ++i, alpha += alpha_delta)
		{
			Matrix4d rotationY;
			rotationY = rotationZ * Affine3d(AngleAxisd(alpha, Vector3d(0, 1, 0))).matrix();

			Matrix4f externalModelMatrix = helmet->renderParams.modelView * rotationY.cast<float>();

			boltSprite.animationIndex = animationIndex[i][p];
			boltSprite.draw(externalModelMatrix, helmet->renderParams.projection);
		}
	}
}

void FxTeslaHelmetOKC::unload()
{
	FX::unload();

	boltSprite.unload();
}