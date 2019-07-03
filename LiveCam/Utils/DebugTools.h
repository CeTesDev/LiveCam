#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <windows.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <Common/Resolutions.h>
#include <Tracking/TrackingTarget.h>
#include <Widgets/FaceMapping.h>

class DebugTools
{
public:
	DebugTools();
	DebugTools::~DebugTools();
#ifdef SHOW_DEBUG_OPENGL
	static void InitDebugConsole();
#endif
	static void ShowTrackingTargets(cv::Mat *output, std::vector<TrackingTarget>& faces);
	static void ShowTargetDetails(cv::Mat *output, std::vector<TrackingTarget>& faces);
	static void ShowContours(cv::Mat *output, cv::Mat &input);
	static void ShowFacialVertices(cv::Mat *output);

};