#pragma once

#include <Tracking/TrackingInterface.h>

class CaffeFaceDetector : public TrackingInterface
{
public:
	CaffeFaceDetector();
	~CaffeFaceDetector();

	void start() override;
	void stop() override;
	void setRefFrame(cv::Mat frame) override;
	cv::Mat preProcess(cv::Mat frame) override;
	std::vector<cv::Rect> scan(cv::Mat frame) override;
	void getPoints(std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> &points) override;
	void scale(double newscale) override;

protected:
	cv::dnn::Net trackingNet;

private:
	bool inputSwapRB = false;
	bool inputCrop = false;
	cv::Size inputSize = cv::Size(300, 300);
	cv::Scalar inputMean = cv::Scalar(0);
	cv::String inputName;
	cv::String outputName;
	std::vector<cv::String> outputNames;

};
