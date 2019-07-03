#pragma once

#include <Tracking/TargetDetailInterface.h>
//FaceDetailScanner

class FaceDetailScanner : public TargetDetailInterface
{
public:
	FaceDetailScanner(bool use3D);
	~FaceDetailScanner();

	cv::Mat preProcess(cv::Mat frame) override;
	cv::Mat preProcessForOneFace(cv::Mat frame, cv::Rect rect);
	void scan(TargetHoldingStruct &face, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& landmarks) override;
	void scale(double newscale) override;

protected:
	bool use3D;

};