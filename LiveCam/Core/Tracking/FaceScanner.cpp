#include "FaceScanner.h"

#include <Common/CommonClasses.h>

FaceScanner::FaceScanner(TRACKER_LIB tLib)
{
	capacity = MAX_TARGETS_COUNT;
	scanConfig.swapRB = true;

	switch (tLib)
	{
	case TRACKER_LIB::YOLO:
		scanConfig.size = cv::Size(416, 416);
		scanConfig.inputName = "";
		scanConfig.outputName = "";
		canPreProcess = false;
		scanNet = cv::dnn::readNetFromDarknet(YOLO_FACE_FILE_1, YOLO_FACE_FILE_2);

		keypointsConfig.size = cv::Size(60, 60);
		keypointsConfig.outputName = "Dense3";
		keypointNet = cv::dnn::readNetFromCaffe(CAFFE_KEYPOINTS_FILE_1, CAFFE_KEYPOINTS_FILE_2);
		break;
	case TRACKER_LIB::TENSORFLOW:
		scanConfig.size = cv::Size(300, 300);
		scanConfig.mean = cv::Scalar(104.0, 177.0, 123.0);
		scanConfig.inputName = "data";
		scanConfig.outputName = "detection_out";
		confidenceThresh = 0.7f;
		scanNet = cv::dnn::readNetFromTensorflow(TENSORFLOW_FACE_FILE_1, TENSORFLOW_FACE_FILE_2);

		keypointsConfig.size = cv::Size(60, 60);
		keypointsConfig.outputName = "Dense3";
		keypointNet = cv::dnn::readNetFromCaffe(CAFFE_KEYPOINTS_FILE_1, CAFFE_KEYPOINTS_FILE_2);
		break;
	default:
		scanConfig.size = cv::Size(300, 300);
		scanConfig.mean = cv::Scalar(104.0, 177.0, 123.0);
		scanConfig.inputName = "data";
		scanConfig.outputName = "detection_out";
		confidenceThresh = 0.8f;
		scanNet = cv::dnn::readNetFromCaffe(CAFFE_FACE_FILE_1, CAFFE_FACE_FILE_2);

		keypointsConfig.size = cv::Size(60, 60);
		keypointsConfig.outputName = "Dense3";
		keypointNet = cv::dnn::readNetFromCaffe(CAFFE_KEYPOINTS_FILE_1, CAFFE_KEYPOINTS_FILE_2);
	}
	//trackingNet.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
	//trackingNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

FaceScanner::~FaceScanner()
{
}

void FaceScanner::start()
{}

void FaceScanner::stop()
{}

void FaceScanner::setRefFrame(cv::Mat frame, int x, int y)
{}

cv::Mat FaceScanner::prepScanFrame(cv::Mat frame)
{
	if (canPreProcess && frame.channels() == 4)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_BGRA2BGR);
		return gray;
	}
	return frame;
}

cv::Mat FaceScanner::prepKeypointsFrame(cv::Mat frame)
{
	if (canPreProcess && frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		return gray;
	}
	return frame;
}

std::vector<cv::Rect> FaceScanner::scan(cv::Mat frame)
{
	cv::Mat resizeFrame;
	std::vector<cv::Rect> results;
	if (!scanNet.empty())
	{
		cv::resize(frame, resizeFrame, scanConfig.size);
		cv::Mat inputBlob = cv::dnn::blobFromImage(frame, activescale, scanConfig.size, scanConfig.mean, scanConfig.swapRB, scanConfig.crop);
		if (!scanConfig.inputName.empty())
		{
			scanNet.setInput(inputBlob, scanConfig.inputName);
		}
		else
		{
			scanNet.setInput(inputBlob);
		}
		cv::Mat detection;
		if (!scanConfig.outputName.empty())
		{
			detection = scanNet.forward(scanConfig.outputName);
		}
		else
		{
			detection = scanNet.forward();
		}
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

void FaceScanner::scale(double newscale)
{
	activescale = newscale;
}

void FaceScanner::keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data)
{
	cv::Rect newRect = target.rect;

	constexpr float cx = 1.3;
	constexpr float cy = 0.85;

	int diffX = newRect.width*cx - newRect.width;
	int diffY = newRect.height*cy - newRect.height;

	newRect.width *= cx;
	newRect.height *= 1.0;
	newRect.x -= diffX * 0.5;
	newRect.y -= diffY;

	target.lastKnownRect = target.rect;
	target.rect = newRect;

	if (newRect.x < 0 || newRect.x + newRect.width >= Resolutions::INPUT_ACTUAL_WIDTH
		|| newRect.y < 0 || newRect.y + newRect.height >= Resolutions::INPUT_ACTUAL_HEIGHT)
	{
		target.visible = false;
		return;
	}

	cv::Mat srcROI(target.frame, newRect);

	cv::Mat gray;
	srcROI.convertTo(gray, CV_32FC1);

	cv::Size preSize(40, 40);
	cv::Mat preprocessed = cv::Mat(preSize, CV_32FC1);
	cv::resize(gray, preprocessed, preSize, 0, 0, cv::INTER_CUBIC);

	cv::Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	cv::meanStdDev(preprocessed, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);

	preprocessed = (preprocessed - m) / (0.000001 + sd);

	cv::Mat landmarkInputBlob = cv::dnn::blobFromImage(preprocessed, activescale, preSize, keypointsConfig.mean, keypointsConfig.swapRB, keypointsConfig.crop);

	landmarkInputBlob.reshape(0, std::vector<int>{ 1, 1, preprocessed.rows, preprocessed.cols });

	keypointNet.setInput(landmarkInputBlob);

	cv::Mat featureBlob = keypointNet.forward(keypointsConfig.outputName);
	size_t maxCols = std::min((featureBlob.cols / 2), TARGET_DETAIL_MODIFIER);
	for (int i = 0; i < maxCols; i++)
	{
		data[i].x = int(featureBlob.at<float>(0, 2 * i) * newRect.width + newRect.x);
		data[i].y = int(featureBlob.at<float>(0, 2 * i + 1) * newRect.height + newRect.y);
	}

}
