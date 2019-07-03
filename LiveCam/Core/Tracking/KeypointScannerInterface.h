#pragma once

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <Tracking/TrackingStructs.h>

class KeypointScannerInterface
{
public:
	virtual cv::Mat preProcess(cv::Mat frame) = 0;
    virtual void scan(TargetHoldingStruct &face, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> &landmarks) = 0;
	virtual void scale(double newscale) = 0;

	cv::Size detailSize = cv::Size(60, 60);
	cv::Scalar detailMean = cv::Scalar(0);
	cv::String detailInputName;
	cv::String detailOutputName;

	bool detailSwapRB = false;
	bool detailCrop = false;

	double activescale = 1.0;

protected:
	cv::dnn::Net detailNet;

};
