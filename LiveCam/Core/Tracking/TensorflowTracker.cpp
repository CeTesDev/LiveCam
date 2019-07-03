#include "TensorflowTracker.h"

#include <Common/CommonClasses.h>

TensorflowTracker::TensorflowTracker()
{
	capacity = MAX_TARGETS_COUNT;
	inputSize = cv::Size(300, 300);
	inputMean = cv::Scalar(104.0, 177.0, 123.0);
	inputName = "data";
	outputName = "detection_out";
	inputSwapRB = true;

	const cv::String trackerGraph = "Assets/model/tensorflow/opencv_face_detector_uint8.pb";
	const cv::String trackerConfiguration = "Assets/model/tensorflow/opencv_face_detector.pbtxt";

	trackingNet = cv::dnn::readNetFromTensorflow(trackerGraph, trackerConfiguration);
	//trackingNet.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
	//trackingNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

TensorflowTracker::~TensorflowTracker()
{
}

void TensorflowTracker::start()
{}

void TensorflowTracker::stop()
{}

void TensorflowTracker::setRefFrame(cv::Mat frame)
{}

cv::Mat TensorflowTracker::preProcess(cv::Mat frame)
{
	if (frame.channels() == 4)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_BGRA2BGR);
		return gray;
	}
	return frame;
}

std::vector<cv::Rect> TensorflowTracker::scan(cv::Mat frame)
{
	cv::Mat resizeFrame;
	cv::resize(frame, resizeFrame, inputSize);
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, activescale, inputSize, inputMean, inputSwapRB, inputCrop);
	trackingNet.setInput(inputBlob);
	cv::Mat detection = trackingNet.forward();
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	std::vector<cv::Rect> faces;

	float confidenceThresh = 0.7f;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThresh)
		{
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

			cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			faces.push_back(object);
		}
	}

	return faces;
}

void TensorflowTracker::getPoints(std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> &points)
{
}

void TensorflowTracker::scale(double newscale)
{
	activescale = newscale;
}
