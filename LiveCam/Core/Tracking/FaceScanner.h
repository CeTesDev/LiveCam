#pragma once

#include <Tracking/ScannerInterface.h>

class FaceScanner : public ScannerInterface
{
public:
	FaceScanner(TRACKER_LIB tLib);
	~FaceScanner();

	/* --ScannerInterface-- */
	void start() override;
	void stop() override;
	void setRefFrame(cv::Mat frame, int x, int y) override;
	cv::Mat prepScanFrame(cv::Mat frame) override;
	cv::Mat prepKeypointsFrame(cv::Mat frame) override;
	std::vector<cv::Rect> scan(cv::Mat frame) override;
	void keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data) override;
	void scale(double newscale) override;

protected:
	cv::dnn::Net scanNet;
	cv::dnn::Net keypointNet;

private:
	TrackerConfigStruct scanConfig;
	TrackerConfigStruct keypointsConfig;
};
