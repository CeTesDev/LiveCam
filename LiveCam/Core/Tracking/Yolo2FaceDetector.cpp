#include "Yolo2FaceDetector.h"

#include <Common/CommonClasses.h>

Yolo2FaceDetector::Yolo2FaceDetector()
{
	capacity = MAX_TARGETS_COUNT;
	inputSize = cv::Size(416, 416);
	inputSwapRB = true;

	cv::String modelConfiguration = "Assets/model/yolo/tiny-yolo-azface-fddb.cfg";
	cv::String modelBinary = "Assets/model/yolo/tiny-yolo-azface-fddb_82000.weights";
	trackingNet = cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary);
	//trackingNet.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
	//trackingNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

Yolo2FaceDetector::~Yolo2FaceDetector()
{
}

void Yolo2FaceDetector::start()
{}

void Yolo2FaceDetector::stop()
{}

void Yolo2FaceDetector::setRefFrame(cv::Mat frame)
{}

cv::Mat Yolo2FaceDetector::preProcess(cv::Mat frame)
{
	/*cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
	cvtColor(frame, gray, cv::COLOR_BGRA2BGR);

	return gray;*/
	return frame;
}

std::vector<cv::Rect> Yolo2FaceDetector::scan(cv::Mat frame)
{
	cv::Mat resizeFrame;
	cv::resize(frame, resizeFrame, inputSize);
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, activescale, inputSize, inputMean, inputSwapRB, inputCrop);
	trackingNet.setInput(inputBlob, inputName);
	cv::Mat detection = trackingNet.forward(outputName);
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	std::vector<cv::Rect> faces;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

		size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

		if (confidence > CONFIDENCE_THRESHOLD)
		{
			float x_center = detectionMat.at<float>(i, 0) * frame.cols;
			float y_center = detectionMat.at<float>(i, 1) * frame.rows;
			float width = detectionMat.at<float>(i, 2) * frame.cols;
			float height = detectionMat.at<float>(i, 3) * frame.rows;
			cv::Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
			cv::Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));


			cv::Rect object(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

			faces.push_back(object);
		}
	}

	return faces;
}

void Yolo2FaceDetector::getPoints(std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> &points)
{
}

void Yolo2FaceDetector::scale(double newscale)
{
	activescale = newscale;
}
