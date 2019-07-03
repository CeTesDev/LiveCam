#pragma once

#ifdef LC_INCLUDE_HAND_TRACKING

#include <Tracking/ScannerInterface.h>

class HandScanner : public ScannerInterface
{
public:
	HandScanner();
	~HandScanner();

	int scanCounter = 0;
	int delayCounter = 0;

	bool pythonInitialized = false;
	bool available = false;

	std::vector<float> pyArrayCopy;

	std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> detectedPoints;

	void asyncScan();
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
	cv::dnn::Net trackingNet;
	std::mutex frameMutex;
	std::mutex scanMutex;

private:
	bool inputSwapRB = false;
	bool inputCrop = false;

	cv::Size inputSize = cv::Size(300, 300);
	cv::Scalar inputMean = cv::Scalar(0);
	cv::String inputName;
	cv::String outputName;
	std::vector<cv::String> outputNames;
};

#endif
