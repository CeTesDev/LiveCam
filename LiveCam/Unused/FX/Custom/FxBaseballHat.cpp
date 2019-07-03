#include <fx/FxBaseballHat.h>

std::array<float, 9> LogoLeftTransformMatrix = {
	-75.0355, 13.8284, -20.7426,
	8.40451, 129.912, 2.90409,
	41.2754, -49.6504, 1
};

std::array<float, 9> LogoRightTransformMatrix = {
	2.43, 0, -0.785973,
	0, -5.00271, 0,
	-1.51248, 1.63664, 1
};

FxBaseballHat::FxBaseballHat()
{
	if (shaderManager == NULL)
		shaderManager = new cwc::glShaderManager;

	shader = shaderManagerWrapper.LoadFromFile("./assets/shaders/vertex/extendedVertexShader.txt", "./assets/shaders/fragment/special/baseballHat_FragmentShader.txt");

	headModel = make_shared<FxWidget3D>();
	headModel->modelPath = "./assets/fx/baseballHat/models/BaseballCap_V3.obj";
	headModel->name = "hat";

	models.push_back(headModel);
}

FxBaseballHat::~FxBaseballHat()
{

}

void FxBaseballHat::load()
{
	FX::load();

	std::string extraTextureFile = "./assets/fx/baseballHat/textures/logo2.png";
	std::string normalMapFile = "./assets/fx/baseballHat/textures/Baseball Cap Normal Small.png";
	std::string hatTextureFile = "./assets/fx/baseballHat/textures/Baseball Cap Diffuse Small.jpg";

	hatTexture = resourceManager.loadTexture(hatTextureFile).ID;
	extraTexture = resourceManager.loadTexture(extraTextureFile).ID;
	normalMap = resourceManager.loadTexture(normalMapFile).ID;
}

void FxBaseballHat::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	double h = 30.0;
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
	headModel->renderParams.projection = projection;
	auto xCenterRawSmooth = headModel->renderParams.xCenterSmoother[face.pointId].smooth(face.xCenterRaw);
	auto yCenterRawSmooth = headModel->renderParams.yCenterSmoother[face.pointId].smooth(face.yCenterRaw);
	auto rollSmooth = headModel->renderParams.rollSmoother[face.pointId].smooth(face.roll);
	auto pitchSmooth = headModel->renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
	auto yawSmooth = headModel->renderParams.yawSmoother[face.pointId].smooth(face.yaw);
	auto widthRawSmooth = headModel->renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);
	double tx = xCenterRawSmooth / face.frameWidth;
	double ty = yCenterRawSmooth / face.frameHeight;
	double hAngleHalf = atan(tan(vAngle / 2) * aspect);
	double planeH = 2.f *h * tan(vAngle / 2);
	double planeW = planeH * aspect;
	double xShift = (tx - 0.5) * planeW;
	double yShift = -(ty - 0.5) * planeH;
	Matrix4d modelMatrix;
	modelMatrix = Affine3d(Translation3d(Vector3d(xShift, yShift, 0))).matrix();

	Vector3d rotateVector = Vector3d(0.0, 0.0, -h).cross(Vector3d(xShift, yShift, -h));
	rotateVector.normalize();
	double rotateAngle = atan((sqrt(xShift*xShift + yShift*yShift)) / h);
	Affine3d rotation(AngleAxisd(rotateAngle, rotateVector));
	rotation *= AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));
	modelMatrix *= rotation.matrix();
	const double widthToDepth = 1.0;
	double width = planeW * widthRawSmooth / face.frameWidth;
	double faceDepth = widthToDepth * width;
	modelMatrix *= Affine3d(Translation3d(0, 0, -faceDepth + 6)).matrix();

	const double scaleK = 0.08;
	double scale = width * scaleK;
	modelMatrix *= Affine3d(Scaling(scale, scale, scale)).matrix();
	modelMatrix = Affine3d(Translation3d(0, 0, -h)).matrix() * modelMatrix;
	headModel->depth = -faceDepth * scale - h;
	headModel->renderParams.modelView = modelMatrix.cast<float>();
	headModel->renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();
	headModel->renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
}

void FxBaseballHat::draw(FXModel& face, ExternalRenderParams &externalRenderParams) {
	const std::vector<GraphicsData>& drawObjects = headModel->objects;

	glPolygonMode(GL_FRONT, GL_FILL);
	glPolygonMode(GL_BACK, GL_FILL);

	glEnable(GL_POLYGON_OFFSET_FILL);

	GLfloat lightpos[] = { 1.5*headModel->bmax[0], 1.5*headModel->bmax[1], 2.5 * headModel->bmax[2], 0.0 };

	GLfloat global_diff[] = { 0, 0.2, 0.4, 1 };
	GLfloat global_spec[] = { 0.8, 0.8, 0.8, 1 };
	GLfloat global_amb[] = { 0.3, 0.3, 0.3, 1 };

	glPolygonOffset(1.0, 1.0);
	GLsizei stride = (3 + 3 + 3 + 2) * sizeof(float);
	if (shader) shader->begin();

	shader->setUniformMatrix3fv("LogoLeftTransformMatrix", 1, false, &LogoLeftTransformMatrix[0]);
	shader->setUniformMatrix3fv("LogoRightTransformMatrix", 1, false, &LogoRightTransformMatrix[0]);

	shader->setUniform3f("cameraPos", 0, 1.f, 1.f);
	shader->setUniform3f("specularLight", 1.f, 1.f, 1.f);
	shader->setUniform3f("ambientLight", 0.1f, 0.1f, 0.1f);

	shader->setUniform4f("lightPos", lightpos[0], lightpos[1], lightpos[2], 1.0);

	for (size_t i = 0; i < drawObjects.size(); i++) {
		GraphicsData o = drawObjects[i];
		if (o.vb < 1) {
			continue;
		}
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		int matId = o.material_id;

		GLfloat material_diff[] = { 0.8, 0.8, 0.8, 1 };
		GLfloat material_spec[] = { 0.5, 0.6, 0.4, 1 };
		GLfloat material_amb[] = { 1, 1, 1, 1 };
		GLfloat shininess = 50.0f;
		GLuint textureID = 0;
		bool hasTexture = false;

		textureID = hatTexture;
		hasTexture = true;

		headModel->renderParams.SetUniforms(*shader);
		
		int textureUniform = shader->GetUniformLocation("Texture");
		int additionalTextureUniform = shader->GetUniformLocation("AdditionalTexture");
		int normalTextureUniform = shader->GetUniformLocation("NormalMap");

		glUniform1i(textureUniform, 0);
		glUniform1i(additionalTextureUniform, 1);

		glUniform1i(normalTextureUniform, 2);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, normalMap);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, extraTexture);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureID);

		glBindBuffer(GL_ARRAY_BUFFER, o.vb);
		int loc = shader->getVetrexShaderAttribLocation("VertexPosition");
		if (loc >= 0)
		{
			glEnableVertexAttribArray(loc);
			shader->BindAttribLocation(loc, "VertexPosition");
			glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
		}

		glBindBuffer(GL_ARRAY_BUFFER, o.nb);

		loc = shader->getVetrexShaderAttribLocation("VertexNormal");
		if (loc >= 0)
		{
			glEnableVertexAttribArray(loc);
			shader->BindAttribLocation(loc, "VertexNormal");
			glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
		}

		glBindBuffer(GL_ARRAY_BUFFER, o.tangentb);
		loc = shader->getVetrexShaderAttribLocation("VertexTangent");
		if (loc >= 0)
		{
			glEnableVertexAttribArray(loc);
			shader->BindAttribLocation(loc, "VertexTangent");
			glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
		}

		glBindBuffer(GL_ARRAY_BUFFER, o.bitangentb);
		loc = shader->getVetrexShaderAttribLocation("VertexBitangent");
		if (loc >= 0)
		{
			glEnableVertexAttribArray(loc);
			shader->BindAttribLocation(loc, "VertexBitangent");
			glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
		}

		if (o.tb > 0 && hasTexture)
		{
				loc = shader->getVetrexShaderAttribLocation("VertexTexCoord");
				if (loc >= 0)
				{
					glEnableVertexAttribArray(loc);
					glEnableClientState(GL_TEXTURE_COORD_ARRAY);
					glBindBuffer(GL_ARRAY_BUFFER, o.tb);
					glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);
				}
	
		}

		glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
		glBindTexture(GL_TEXTURE_2D, 0);

		glPopAttrib();
	}
	if (shader) shader->end();
}