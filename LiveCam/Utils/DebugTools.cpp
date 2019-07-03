#include "DebugTools.h"

cv::RNG rng(12345);

#ifdef SHOW_DEBUG_OPENGL

void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

void DebugTools::InitDebugConsole()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	GLint flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
	if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(glDebugOutput, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}
}
#endif

void DebugTools::ShowTrackingTargets(cv::Mat *output, std::vector<TrackingTarget>& faces)
{
	cv::Scalar c_green(0, 255, 0);
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces[i].inited)
		{
			cv::Rect rect = faces[i].rect;
			float xMod = (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			float yMod = (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;
			float xOut = rect.x * xMod * 1.28;
			float yOut = rect.y * yMod * 1.96;
			float widthOut = rect.width * xMod * 0.48;
			float heightOut = rect.height * yMod * 0.56;
			cv::ellipse((*output), cv::Point2f(xOut, yOut), cv::Size(widthOut, heightOut), 0, 0, 360, c_green, 4, 8, 0);
		}
	}
}

void DebugTools::ShowTargetDetails(cv::Mat *output, std::vector<TrackingTarget>& faces)
{
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 2;
	int baseline = 0;

	cv::Scalar c_white(255, 255, 255);
	cv::Scalar c_grey(200, 200, 200);
	cv::Scalar textColor;
	bool nextIsWhite = false;

	for (size_t i = 0; i < faces.size(); i++)
	{
		if (faces[i].inited)
		{
			cv::Rect rect = faces[i].rect;

			float xMod = (float)(Resolutions::OUTPUT_WIDTH) / faces[i].frameWidth;
			float yMod = (float)(Resolutions::OUTPUT_HEIGHT) / faces[i].frameHeight;
			/*
			float xOut = rect.x * xMod * 1.28;
			float yOut = rect.y * yMod * 1.96;
			float widthOut = rect.width * xMod * 0.48;
			float heightOut = rect.height * yMod * 0.56;

			cv::ellipse(output, cv::Point2f(xOut, yOut), cv::Size(widthOut, heightOut), 0, 0, 360, cv::Scalar(0, 255, 0), 4, 8, 0);
			*/
			for (size_t j = 0; j < TARGET_DETAIL_LIMIT; j++)
			{
				cv::Point2f actual = cv::Point2f(faces[i].pts[2 * j], faces[i].pts[2 * j + 1]);
				cv::Point2f p = cv::Point2f(actual.x * xMod, actual.y * yMod);

				float blueValue = 255.0f;
				float greenValue = 0.0f;
				float redValue = 0.0f;
				if (j > 47) { //MOUTH 48-66
					blueValue = 0;
					greenValue = 0;
					redValue = 255;
				}
				else if (j > 41) { //RIGHT EYE 42-47
					blueValue = 0;
					greenValue = 128;
					redValue = 255;
				}
				else if (j > 35) { //LEFT EYE 36-41
					blueValue = 0;
					greenValue = 255;
					redValue = 255;
				}
				else if (j > 26) {
					blueValue = 0;
					greenValue = 255;
					redValue = 0;
				}
				cv::Scalar color(blueValue, greenValue, redValue, 0.33f);
				cv::ellipse((*output), p, cv::Size(2, 2), 0, 0, 360, color, 4, 8, 0);
				textColor = (nextIsWhite) ? c_white : c_grey;
				std::string text = std::to_string(j);
				cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
				float tX;
				float tY = p.y;
				if (j == 60)
				{
					tX = (p.x - (textSize.width * 1.2));
				}
				else if (j == 54) {
					tX = (p.x + (textSize.width * 1.2));
				}
				else {
					tX = (p.x - (textSize.width / 2));
					tY = (p.y + (textSize.height * 2));
				}
				cv::Point textOrg(tX, tY);

				cv::putText((*output), text, textOrg, fontFace, fontScale, textColor, thickness, cv::LINE_AA);
				if (j > 60)
				{
					std::ostringstream pointInfo;
					pointInfo << j << ": " << actual.x << ", " << actual.y << " (" << p.x << ", " << p.y << ")";
					std::string infoText = pointInfo.str();
					cv::Point infoOrg(textSize.height, (textSize.height + ((textSize.height * 1.5) * (j - 60))));
					cv::putText((*output), infoText, infoOrg, fontFace, fontScale, color, thickness, cv::LINE_AA);
					nextIsWhite = !nextIsWhite;
				}
			}
		}
	}
}

void DebugTools::ShowContours(cv::Mat *output, cv::Mat &input)
{
	cv::Mat gray;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	cv::blur(gray, gray, cv::Size(3, 3));
	cv::Mat canny_output;
	cv::Canny(gray, canny_output, 100, 200, 3);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::drawContours((*output), contours, i, color, 2, 8, hierarchy, 0, cv::Point());
	}
}

void DebugTools::ShowFacialVertices(cv::Mat *output)
{
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 2;
	int baseline = 0;
	cv::Scalar c_white(255, 255, 255);
	for (size_t i = 0; i < TARGET_DETAIL_LIMIT; i++)
	{
		cv::Point2f p = cv::Point2f(FaceMapping::InitTable[i].x, FaceMapping::InitTable[i].y);
		//cv::ellipse((*output), p, cv::Size(2, 2), 0, 0, 360, c_white, 4, 8, 0);
		std::string text = std::to_string(i);
		cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
		cv::Point textOrg((p.x - (textSize.width / 2)), (p.y + (textSize.height * 2)));
		cv::putText((*output), text, textOrg, fontFace, fontScale, c_white, thickness, cv::LINE_AA);
	}
}

