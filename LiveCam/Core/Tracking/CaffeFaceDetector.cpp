#include "CaffeFaceDetector.h"

#include <Common/CommonClasses.h>

CaffeFaceDetector::CaffeFaceDetector()
{
	capacity = MAX_TARGETS_COUNT;
	inputSize = cv::Size(300, 300);
	inputMean = cv::Scalar(104.0, 177.0, 123.0);
	inputName = "data";
	outputName = "detection_out";

	const cv::String trackerConfiguration = "Assets/model/caffe/deploy.prototxt";
	const cv::String trackerBinary = "Assets/model/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel";

	trackingNet = cv::dnn::readNetFromCaffe(trackerConfiguration, trackerBinary);
	//trackingNet.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
	//trackingNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

CaffeFaceDetector::~CaffeFaceDetector()
{
}

void CaffeFaceDetector::start()
{}

void CaffeFaceDetector::stop()
{}

void CaffeFaceDetector::setRefFrame(cv::Mat frame)
{}

cv::Mat CaffeFaceDetector::preProcess(cv::Mat frame)
{
	if (frame.channels() == 4)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_BGRA2BGR);
		return gray;
	}
	return frame;
}

std::vector<cv::Rect> CaffeFaceDetector::scan(cv::Mat frame)
{
	cv::Mat resizeFrame;
	std::vector<cv::Rect> results;
	if (!trackingNet.empty())
	{
		cv::resize(frame, resizeFrame, inputSize);
		cv::Mat inputBlob = cv::dnn::blobFromImage(frame, activescale, inputSize, inputMean, inputSwapRB, inputCrop);
		trackingNet.setInput(inputBlob, inputName);
		cv::Mat detection = trackingNet.forward(outputName);
		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > CONFIDENCE_THRESHOLD)
			{
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				results.push_back(object);
			}
		}
	}
	
	return results;
}

void CaffeFaceDetector::getPoints(std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> &points)
{
}

void CaffeFaceDetector::scale(double newscale)
{
	activescale = newscale;
}
