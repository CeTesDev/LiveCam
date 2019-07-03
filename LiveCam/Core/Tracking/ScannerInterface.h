#pragma once

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <Tracking/TrackingStructs.h>

class ScannerInterface
{
public:
	int capacity = 1;

	virtual void start() = 0;
	virtual void stop() = 0;
	virtual void setRefFrame(cv::Mat frame, int x, int y) = 0;
	virtual cv::Mat prepScanFrame(cv::Mat frame) = 0;
	virtual cv::Mat prepKeypointsFrame(cv::Mat frame) = 0;
    virtual std::vector<cv::Rect> scan(cv::Mat frame) = 0;
	virtual void keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data) = 0;
    virtual void scale(double newscale) = 0;

	double activescale = 1.0;
	bool canPreProcess = true;

	float confidenceThresh = CONFIDENCE_THRESHOLD;
};
