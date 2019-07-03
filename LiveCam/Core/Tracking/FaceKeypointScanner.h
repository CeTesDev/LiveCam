#pragma once

#include <Tracking/KeypointScannerInterface.h>

class FaceKeypointScanner : public KeypointScannerInterface
{
public:
	FaceKeypointScanner(bool use3D);
	~FaceKeypointScanner();

	cv::Mat preProcess(cv::Mat frame) override;
	cv::Mat preProcessForOneFace(cv::Mat frame, cv::Rect rect);
	void scan(TargetHoldingStruct &face, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& landmarks) override;
	void scale(double newscale) override;

protected:
	bool use3D;

};
