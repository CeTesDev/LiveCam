#pragma once
#define COMPILER_MSVC
#define NOMINMAX

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include <detectors/DetectorTypes.h>

#include "ValueSmoother.h"

#include "ObjectDetectorInterface.h"
#include "LandmarkDetectorInterface.h"

#ifdef USE_VLAD_TRACKER


class HandLandmarkDetector : public LandmarkDetectorInterface
{
public:
	HandLandmarkDetector();

	cv::Mat preprocessEntireFrame(cv::Mat frame) override; // for all faces
	cv::Mat preprocessTargetFrame(cv::Mat frame, cv::Rect faceRect); // for one face
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks) override;

protected:

	cv::Mat frameCopy;
};




#endif
