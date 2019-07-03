#include "VulkanMain.h"

#define _USE_MATH_DEFINES
#include <math.h>

double cameraDefaultVerticalAngle = M_PI / 4;
double VulkanzNear = 0.001;
double VulkanzFar = 100;

GLuint cameraTextureId = 0;

const std::map<std::string, AntiAliasing> AntiAliasingMap =
{
	{ "NONE", AntiAliasing::NONE },
	{ "SSAA_4X", AntiAliasing::SSAA_4X }
};

void VulkanMain::drawForeground(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

 	glEnable(GL_TEXTURE_2D);
 	glEnable(GL_DEPTH_TEST);
 	glDepthMask(GL_TRUE);

 	glEnable(GL_CULL_FACE);
 	glCullFace(GL_BACK);
 	glFrontFace(GL_CCW);

	auto thisMoment = std::chrono::system_clock::now();
	static auto lastMoment = thisMoment;
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(thisMoment - lastMoment).count() / 1000.f;
	lastMoment = thisMoment;

	if (millis > 0.050f)
	{
		millis = 0.050f;
	}

 	ExternalRenderParams params =
 	{
 		frame_width, frame_height, frame.cols, frame.rows, cameraDefaultVerticalAngle, VulkanzNear, VulkanzFar, millis
 	};

	if (currentEffect->useFrameLightParams)
	{
		params.frameParams = CVImageUtil::getFrameLightParams(frame);
	}

  	for (auto &face : faces)
  	{
		if (currentEffect->useFaceLightParams)
		{
			auto contours = CVImageUtil::extractFaceContourInFrame(face);
			if (contours.size.width > 0 && contours.size.height > 0)
			{
				params.faceParams = CVImageUtil::getFaceLightParams(frame, contours, face);
			}
		}

		currentEffect->outerTransform(opencvFaces, face, params);
		currentEffect->outerDraw(face, params);
  	}

	glPopAttrib();
}

cv::Point2f fixPointPosition(cv::Point2f A, cv::Point2f B, cv::Point2f P)
{

	cv::Point2f AP = P - A;
	cv::Point2f AB = B - A;
	float lenAB = sqrtf(AB.x*AB.x + AB.y * AB.y);

	float lenAP = sqrtf(AP.x*AP.x + AP.y * AP.y);

	float cosA = (AP.x *AB.x + AP.y * AB.y) / (lenAB * lenAP);

	float lenAK = lenAP * cosA;

	cv::Point2f K = A + AB * lenAK/ lenAB;

	cv::Point2f KP = P - K;

	float lenKP = sqrtf(KP.x*KP.x + KP.y * KP.y);

	static const float percent = 0.05;

	if (lenKP > percent * lenAB)
	{
		float lenKPfixed = percent * lenAB;

		cv::Point2f Pfixed = K + lenKPfixed * KP / lenKP;

		return Pfixed;
	}

	return P;
	
}

cv::Point2f fixChinPosition(cv::Point2f A, cv::Point2f B, cv::Point2f P)
{

	cv::Point2f AP = P - A;
	cv::Point2f AB = B - A;
	float lenAB = sqrtf(AB.x*AB.x + AB.y * AB.y);

	float lenAP = sqrtf(AP.x*AP.x + AP.y * AP.y);

	float cosA = (AP.x *AB.x + AP.y * AB.y) / (lenAB * lenAP);

	float lenAK = lenAP * cosA;

	cv::Point2f K = A + AB * lenAK / lenAB;

	cv::Point2f KP = P - K;

	float lenKP = sqrtf(KP.x*KP.x + KP.y * KP.y);

	static const float percent = 0.7;

	if (lenKP > percent * lenAB)
	{
		float lenKPfixed = percent * lenAB;

		cv::Point2f Pfixed = K + lenKPfixed * KP / lenKP;

		return Pfixed;
	}

	return P;

}

cv::Mat VulkanMain::drawBackground(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces)
{
	if (currentEffect != nullptr)
	{
		for (auto &face : faces)
		{
			currentEffect->drawOnCVFrame_OLD(frame, face);
		}
		
		currentEffect->transformMesh(frame, faces, &mesh);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, cameraTextureId);

		
	ResourceManager::createTexture(frame);

	frameManager.SwitchToFrameBuffer(cameraFrameIndex);

	static cv::Mat returnImg(frame_height, frame_width, CV_8UC3);

	if (mesh.vertices.size() > 0) 
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		textureShader->begin();
		GLenum texEnum = GL_TEXTURE0;
		GLuint texIndex = 0;
		glDepthMask(GL_FALSE);
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_BLEND);
		glActiveTexture(texEnum);		

		glBindTexture(GL_TEXTURE_2D, cameraTextureId);
		glUniform1i(textureShader->GetUniformLocation("Texture"), texIndex);

		textureShader->setUniformMatrix4fv("ProjectionMatrix", 1, false, textureProjection.data());
		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*textureShader, textureQuad.vb));
		VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*textureShader, textureQuad.tb));

		glDrawArrays(GL_TRIANGLES, 0, 6);

		textureShader->end();
		
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glDisable(GL_CULL_FACE);

		glDisable(GL_LIGHTING);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, frame_width, frame_height, 0, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		for (int k = 0; k < mesh.vertices.size(); k++) {
			mesh.vertices[k].x *= frame_width;
			mesh.vertices[k].y *= frame_height;
		}
		int rows = mesh.rows;
		int cols = mesh.cols;
		glBegin(GL_QUADS);
		for (int ir = 0; ir < rows; ir++)
		{
			for (int ic = 0; ic < cols; ic++)
			{
				int idx00 = (cols + 1)*ir + ic;
				int idx01 = idx00 + 1;
				int idx10 = idx00 + (cols + 1);
				int idx11 = idx10 + 1;
				glTexCoord2f(mesh.uvs[idx00].x, mesh.uvs[idx00].y); glVertex2f(mesh.vertices[idx00].x, mesh.vertices[idx00].y);
				glTexCoord2f(mesh.uvs[idx01].x, mesh.uvs[idx01].y); glVertex2f(mesh.vertices[idx01].x, mesh.vertices[idx01].y);
				glTexCoord2f(mesh.uvs[idx11].x, mesh.uvs[idx11].y); glVertex2f(mesh.vertices[idx11].x, mesh.vertices[idx11].y);
				glTexCoord2f(mesh.uvs[idx10].x, mesh.uvs[idx10].y); glVertex2f(mesh.vertices[idx10].x, mesh.vertices[idx10].y);
			}
		}
		glEnd();
	}
	else
	{
		DrawTexture(cameraTextureId, textureShader);
	}
	

	glDepthMask(GL_TRUE);
	glClear(GL_DEPTH_BUFFER_BIT);

	if (currentEffect->useAlphaMask)
	{
		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::ALPHA_MASK);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);

		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::FILTER_MASK);
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	drawForeground(opencvFaces, frame, faces);

	if (currentEffect->useAlphaMask)
	{
		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::COLOR_1);
		auto &FBO = frameManager.getFrameRenderBuffer(frameManager.getCurrentFBOindex());
		DrawTextures({ { FBO.filterMaskId, "FilterMask" }, { FBO.alphaMaskId, "AlphaMask" } }, alphaMaskShader);
	}

	currentEffect->drawParticles(particlesProjection);

	PostProcessing();
	glReadPixels(0, 0, returnImg.cols, returnImg.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, returnImg.data);

#ifdef SHOW_DEBUG_FACE_RECT

	
	
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces[i].inited)
		{
			for (size_t j = 0; j < 66; j++)
			{

				cv::Point2f p = cv::Point2f(faces[i].pts[2 * j], faces[i].pts[2 * j + 1]);

				p.x *= (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
				p.y *= (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;

				cv::ellipse(returnImg, p, cv::Size(2, 2), 0, 0, 360, cv::Scalar(255, 128, 128), 4, 8, 0);

			}

		}
	}

	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces[i].inited)
		{
			cv::Rect lastKnownRect = faces[i].lastKnownRect;
			cv::Rect rect = faces[i].rect;

			lastKnownRect.x *= (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			lastKnownRect.y *= (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;
			lastKnownRect.width *= (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			lastKnownRect.height *= (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;

			rect.x *= (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			rect.y *= (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;
			rect.width *= (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			rect.height *= (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;

			cv::rectangle(returnImg, lastKnownRect, cv::Scalar(128, 255, 128), 4, 8, 0);
			cv::rectangle(returnImg, rect, cv::Scalar(255, 128, 128), 4, 8, 0);
		}
	}
	
#endif

	if (!antiAliasingShader)
	{
		cv::flip(returnImg, returnImg, 0);
	}
	
	return returnImg;
}

cv::Mat VulkanMain::addTrackerValues(std::vector<cv::Rect>& opencvFaces, cv::Mat frame, std::vector<FXModel>& faces)
{
	cv::Mat resultFrame;

	if (!faces.empty())
	{
		resultFrame = drawBackground(opencvFaces, frame, faces);
	}
	else
	{
		resultFrame = frame;
	}

	return resultFrame;
}

void VulkanMain::updateAllTextures(FilterModule *module)
{
	if (currentEffect == nullptr) return;

	for (int i = 0; i < ObjectTracker::MAX_TO_TRACK; ++i)
	{
		currentEffect->applyModule(module, i, true);
	}
}

void VulkanMain::updateTexture(FilterModule *module, int targetIndex)
{
	if (currentEffect == nullptr) return;
	currentEffect->applyModule(module, targetIndex, true);
}

bool VulkanMain::loadEffect(FX *effect)
{
	if (currentEffect != nullptr)
	{
		resourceManager.clear();

		currentEffect->unload();
	}

	
	if (effect != nullptr)
	{
		resourceManager.resourceRoot = effect->resourcesRoot;

		effect->load();
	}

	currentEffect = effect;

	return true;
}

void VulkanMain::close() {
	glfwDestroyWindow(window);
}

int VulkanMain::init(AntiAliasing antiAliasing)
{
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW." << std::endl;
		return -1;
	}
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	window = glfwCreateWindow(Resolutions::OUTPUT_WIDTH, Resolutions::OUTPUT_HEIGHT, "", NULL, NULL);
	if (window == NULL) {
		std::cerr << "Failed to open GLFW window. " << std::endl;
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW." << std::endl;
		return -1;
	}

	frame_width = Resolutions::OUTPUT_WIDTH;
	frame_height = Resolutions::OUTPUT_HEIGHT;

	RecreateSimpleTexture(frame_height, frame_height);

	textureShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/texture.vertex", "./Assets/shaders/post-processing/texture.frag");
	
	textureQuad = GraphicsLoader::CreateQuadModel();

	textureProjection = Matrix4f::Zero();
	textureProjection(0, 0) = 1;
	textureProjection(1, 1) = -1;
	textureProjection(2, 2) = -1;
	textureProjection(3, 3) = 1;

	particlesProjection = Matrix4f::Zero();
	particlesProjection(0, 0) = 2.f / frame_width;
	particlesProjection(1, 1) = 2.f / frame_height;
	particlesProjection(2, 2) = -2.f / 10000;
	particlesProjection(3, 3) = 1;

	int FBOfactor;

	switch (antiAliasing)
	{
	case AntiAliasing::SSAA_4X:
		antiAliasingShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/texture.vertex", "./Assets/shaders/post-processing/SSAA_4X.frag");
		cameraFrameIndex = frameManager.AddFrameBuffer(Resolutions::OUTPUT_WIDTH * 2, Resolutions::OUTPUT_HEIGHT * 2);
		FBOfactor = 2;
		initAntiAliasing4XShader();
		break;
	
	case AntiAliasing::NONE:
		antiAliasingShader = nullptr;
		FBOfactor = 1;
		cameraFrameIndex = frameManager.AddFrameBuffer(Resolutions::OUTPUT_WIDTH, Resolutions::OUTPUT_HEIGHT);
		break;
	}

	alphaMaskShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/texture.vertex", "./Assets/shaders/post-processing/alphaMask.frag");

	alphaMaskShader->begin();

	alphaMaskShader->setUniform1i("WinWidth", Resolutions::OUTPUT_WIDTH * FBOfactor);
	alphaMaskShader->setUniform1i("WinHeight", Resolutions::OUTPUT_HEIGHT * FBOfactor);

	alphaMaskShader->end();

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glAlphaFunc(GL_GREATER, 0.05f);

	return 0;
}

void VulkanMain::initAntiAliasing4XShader()
{

	const float cos30 = sqrt(3) / 2;
	const float sin30 = 0.5f;
	const float sampleRadiusX = 0.75 / Resolutions::OUTPUT_WIDTH;
	const float sampleRadiusY = 0.75 / Resolutions::OUTPUT_HEIGHT;

	antiAliasingShader->begin();

	Vector2f offset;
	offset = Vector2f(cos30 * sampleRadiusX, sin30 * sampleRadiusY);
	antiAliasingShader->setUniform2fv("samplesOffset[0]", 1, &offset[0]);
	offset = Vector2f(-sin30 * sampleRadiusX, cos30 * sampleRadiusY);
	antiAliasingShader->setUniform2fv("samplesOffset[1]", 1, &offset[0]);
	offset = Vector2f(-cos30 * sampleRadiusX, -sin30 * sampleRadiusY);
	antiAliasingShader->setUniform2fv("samplesOffset[2]", 1, &offset[0]);
	offset = Vector2f(sin30 * sampleRadiusX, -cos30 * sampleRadiusY);
	antiAliasingShader->setUniform2fv("samplesOffset[3]", 1, &offset[0]);

	antiAliasingShader->end();
}

void VulkanMain::RecreateSimpleTexture(int width, int height)
{
	if (cameraTextureId != 0)
	{
		glDeleteTextures(1, &cameraTextureId);
	}

	glGenTextures(1, &cameraTextureId);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, cameraTextureId);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

}

void VulkanMain::DrawTexture(GLuint texId, std::shared_ptr<cwc::glShader> shader)
{
	DrawTextures({ { texId, "Texture" } }, shader);
}

void VulkanMain::DrawTextures(std::vector<TextureUniform> textures, std::shared_ptr<cwc::glShader> shader)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	shader->begin();

	glDepthMask(GL_FALSE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);

	GLenum texEnum = GL_TEXTURE0;
	GLuint texIndex = 0;

	for (auto &texture : textures)
	{
		glActiveTexture(texEnum);
		glBindTexture(GL_TEXTURE_2D, texture.first);
		glUniform1i(shader->GetUniformLocation(texture.second.c_str()), texIndex);
		++texIndex;
		++texEnum;
	}

	shader->setUniformMatrix4fv("ProjectionMatrix", 1, false, textureProjection.data());

	VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, textureQuad.vb));
	VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*shader, textureQuad.tb));

	glDrawArrays(GL_TRIANGLES, 0, 6);

	shader->end();

	glPopAttrib();
}

#define swapColorBuffer() \
{ \
	std::swap(sourceBuffer, destinyBuffer); \
	frameManager.SwitchToDrawBuffer(destinyBuffer == FBO.textureId ? (size_t)ColorBuffer::COLOR_1 : (size_t)ColorBuffer::COLOR_2); \
}

void VulkanMain::PostProcessing()
{
	auto &FBO = frameManager.getFrameRenderBuffer(frameManager.getCurrentFBOindex());
	GLuint sourceBuffer, destinyBuffer;
	if (frameManager.getCurrentDrawBuffer() == (size_t)ColorBuffer::COLOR_1)
	{
		sourceBuffer = FBO.textureId;
		destinyBuffer = FBO.add_textureId;
	}
	else
	{
		sourceBuffer = FBO.add_textureId;
		destinyBuffer = FBO.textureId;
	}

	if (antiAliasingShader)
	{
		frameManager.SwitchToScreen();
		DrawTexture(sourceBuffer, antiAliasingShader);
	}
}

