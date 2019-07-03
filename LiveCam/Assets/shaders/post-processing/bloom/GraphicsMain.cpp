#include "GraphicsMain.h"

#include <Graphics/GraphicsLoader.h>

#define _USE_MATH_DEFINES
#include <math.h>

double cameraDefaultVerticalAngle = M_PI / 4;
double GraphicszNear = 0.001;
double GraphicszFar = 100;

GLuint cameraTextureId = 0;

const std::map<std::string, AntiAliasing> AntiAliasingMap =
{
	{ "NONE", AntiAliasing::NONE },
	{ "SSAA_4X", AntiAliasing::SSAA_4X }
};

int GraphicsMain::init(AntiAliasing antiAliasing)
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

	modelViewProjection = Matrix4f::Zero();
	modelViewProjection(0, 0) = 1;
	modelViewProjection(1, 1) = -1;
	modelViewProjection(2, 2) = -1;
	modelViewProjection(3, 3) = 1;

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

	lightingShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/bloom/bloom_stage2.vertex", "./Assets/shaders/post-processing/bloom/bloom_stage2.frag");

	blurShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/bloom/bloom_stage3.vertex", "./Assets/shaders/post-processing/bloom/bloom_stage3.frag");

	blurShader->begin();
	blurShader->setUniform1i("Texture", 0);
	blurShader->end();

	bloomShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/post-processing/bloom/bloom_stage4.vertex", "./Assets/shaders/post-processing/bloom/bloom_stage4.frag");

	bloomShader->begin();
	bloomShader->setUniform1i("Texture", 0);
	bloomShader->setUniform1i("BloomTexture", 1);
	bloomShader->end();

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glAlphaFunc(GL_GREATER, 0.05f);

	float quadVertices[] = {
		// positions        // texture Coords
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
	};
	// setup plane VAO
	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

	return 0;
}

void GraphicsMain::renderQuad()
{
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
}

void GraphicsMain::drawForeground(cv::Mat frame, std::vector<TrackingTarget>& faces)
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
 		frame_width, frame_height, frame.cols, frame.rows, cameraDefaultVerticalAngle, GraphicszNear, GraphicszFar, millis
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

		currentEffect->outerTransform(face, params);
		currentEffect->outerDraw(face, params);
  	}

	glPopAttrib();
}

cv::Mat GraphicsMain::drawBackground(cv::Mat frame, std::vector<TrackingTarget>& faces)
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

	cv::Mat returnImg(frame_height, frame_width, CV_8UC3);

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

	drawForeground(frame, faces);

	if (currentEffect->useAlphaMask)
	{
		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::COLOR_1);
		auto &FBO = frameManager.getFrameRenderBuffer(frameManager.getCurrentFBOindex());
		DrawTextures({ { FBO.filterMaskId, "FilterMask" }, { FBO.alphaMaskId, "AlphaMask" } }, alphaMaskShader);
	}

	currentEffect->drawParticles(particlesProjection);

	PostProcessing();
	glReadPixels(0, 0, returnImg.cols, returnImg.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, returnImg.data);

#ifdef SHOW_DEBUG_CONTOURS
	DebugTools::ShowContours(&returnImg, frame);
#endif

#ifdef SHOW_DEBUG_FACE_RECT
	DebugTools::ShowTrackingTargets(&returnImg, faces);
#endif

#ifdef _SHOW_DEBUG_FACE_DATA
	DebugTools::ShowTargetDetails(&returnImg, faces);
#endif

#ifdef SHOW_DEBUG_FACE_VERTICES
	DebugTools::ShowFacialVertices(&returnImg);
#endif

	if (!antiAliasingShader)
	{
		cv::flip(returnImg, returnImg, 0);
	}
	
	return returnImg;
}

cv::Mat GraphicsMain::addTrackerValues(cv::Mat frame, std::vector<TrackingTarget>& faces)
{
	cv::Mat resultFrame;

	if (!faces.empty())
	{
		resultFrame = drawBackground(frame, faces);
	}
	else
	{
		resultFrame = frame;
	}

	return resultFrame;
}

void GraphicsMain::updateAllTextures(FilterModule *module)
{
	if (currentEffect == nullptr) return;

	for (int i = 0; i < MAX_TO_TRACK; ++i)
	{
		currentEffect->applyModule(module, i, true);
	}
}

void GraphicsMain::updateTexture(FilterModule *module, int targetIndex)
{
	if (currentEffect == nullptr) return;
	currentEffect->applyModule(module, targetIndex, true);
}

bool GraphicsMain::loadEffect(FX *effect)
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

void GraphicsMain::close() {
	glfwDestroyWindow(window);
}

void GraphicsMain::initAntiAliasing4XShader()
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

void GraphicsMain::RecreateSimpleTexture(int width, int height)
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

void GraphicsMain::DrawTexture(GLuint texId, std::shared_ptr<cwc::glShader> shader)
{
	DrawTextures({ { texId, "Texture" } }, shader);
}

void GraphicsMain::DrawTextures(std::vector<TextureUniform> textures, std::shared_ptr<cwc::glShader> shader)
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

void GraphicsMain::DrawHDR(std::vector<TextureUniform> textures)
{
	auto &FBO = frameManager.getFrameRenderBuffer();

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	blurShader->begin();

	glDepthMask(GL_FALSE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);

	// Gaussian Blur pass
	bool horizontal = true, first_iteration = true;
	unsigned int amount = 10;
	for (GLuint i = 0; i < amount; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBO.hdrBuffers[horizontal]);
		blurShader->setUniform1i("horizontal", horizontal);
		glBindTexture(GL_TEXTURE_2D, first_iteration ? FBO.texture2Id : FBO.hdrTextures[!horizontal]);
		glBindVertexArray(quadVAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
		horizontal = !horizontal;
		if (first_iteration)
		{
			first_iteration = false;
		}
	}
	VertexAttribSetter v1Position(VertexAttribSetter::PositionAttribSetter(*blurShader, textureQuad.vb));
	VertexAttribSetter v1TexCoord(VertexAttribSetter::TexCoordAttribSetter(*blurShader, textureQuad.tb));
	blurShader->end();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	bloomShader->begin();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, FBO.textureId);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, FBO.hdrTextures[!horizontal]);
	GLenum texEnum = GL_TEXTURE0;
	GLuint texIndex = 0;

	for (auto &texture : textures)
	{
		glActiveTexture(texEnum);
		glBindTexture(GL_TEXTURE_2D, texture.first);
		glUniform1i(bloomShader->GetUniformLocation(texture.second.c_str()), texIndex);
		++texIndex;
		++texEnum;
	}
	bloomShader->setUniformMatrix4fv("ProjectionMatrix", 1, false, textureProjection.data());
	bloomShader->setUniformMatrix4fv("ModelViewMatrix", 1, false, modelViewProjection.data());
	VertexAttribSetter v2Position(VertexAttribSetter::PositionAttribSetter(*bloomShader, textureQuad.vb));
	VertexAttribSetter v2TexCoord(VertexAttribSetter::TexCoordAttribSetter(*bloomShader, textureQuad.tb));
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	bloomShader->end();

	glPopAttrib();
}

void GraphicsMain::PostProcessing()
{
	auto &FBO = frameManager.getFrameRenderBuffer(frameManager.getCurrentFBOindex());
	GLuint sourceBuffer, destinyBuffer;
	if (frameManager.getCurrentDrawBuffer() == (size_t)ColorBuffer::COLOR_1)
	{
		sourceBuffer = FBO.textureId;
		destinyBuffer = FBO.texture2Id;
	}
	else
	{
		sourceBuffer = FBO.texture2Id;
		destinyBuffer = FBO.textureId;
	}

	if (antiAliasingShader)
	{
		frameManager.SwitchToScreen();
		DrawTexture(sourceBuffer, antiAliasingShader);
	}

	if (currentEffect->useAlphaMask)
	{
		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::COLOR_1);
		auto &FBO = frameManager.getFrameRenderBuffer(frameManager.getCurrentFBOindex());
		DrawTextures({ { FBO.filterMaskId, "FilterMask" }, { FBO.alphaMaskId, "AlphaMask" } }, alphaMaskShader);
	}

	if (currentEffect->useBloomLighting)
	{
		frameManager.SwitchToDrawBuffer((size_t)ColorBuffer::COLOR_1);
		DrawHDR({ { sourceBuffer, "Texture" }, { destinyBuffer, "BloomTexture" } });
		//DrawTextures({ { sourceBuffer, "Texture" }, { destinyBuffer, "BloomTexture" } }, bloomShader);
	}
}

void GraphicsMain::SwapColorBuffer(FrameRenderBuffer frb, GLuint sb, GLuint tb)
{
	std::swap(sb, tb);
	frameManager.SwitchToDrawBuffer(tb == frb.textureId ? (size_t)ColorBuffer::COLOR_1 : (size_t)ColorBuffer::COLOR_2);
}

