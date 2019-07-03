#include "FaceDetailScanner.h"
#include <Common/CommonClasses.h>

FaceDetailScanner::FaceDetailScanner(bool use3D)
{
	this->use3D = use3D;

	const cv::String configuration3D = "Assets/model/caffe/fa_deploy.prototxt";
	const cv::String binary3D = "Assets/model/caffe/fa__iter_1400000.caffemodel";

	const cv::String configuration2D = "Assets/model/caffe/landmark_deploy.prototxt";
	const cv::String binary2D = "Assets/model/caffe/VanFace.caffemodel";

	detailNet = use3D ?
		cv::dnn::readNetFromCaffe(configuration3D, binary3D) :
		cv::dnn::readNetFromCaffe(configuration2D, binary2D);

	detailSize = use3D ? cv::Size(40, 40) : cv::Size(60, 60);

	detailOutputName = use3D ? "Dense2" : "Dense3";
}

FaceDetailScanner::~FaceDetailScanner()
{
}

void FaceDetailScanner::scale(double newscale)
{
	activescale = newscale;
}

cv::Mat FaceDetailScanner::preProcess(cv::Mat frame)
{
	if (frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		return gray;
	}
	else
	{
		return frame;
	}
}

cv::Mat FaceDetailScanner::preProcessForOneFace(cv::Mat frame, cv::Rect rect)
{
	cv::Mat srcROI(frame, rect);

	cv::Mat gray;
	srcROI.convertTo(gray, CV_32FC1);

	cv::Size preSize(40, 40);
	cv::Mat smallGray = cv::Mat(preSize, CV_32FC1);
	cv::resize(gray, smallGray, preSize, 0, 0, cv::INTER_CUBIC);

	cv::Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	cv::meanStdDev(smallGray, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);

	smallGray = (smallGray - m) / (0.000001 + sd);

	return smallGray;
}

void FaceDetailScanner::scan(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data)
{
	cv::Rect rect = target.rect;

	cv::Rect newRect = rect;

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

	cv::Mat preprocessed = preProcessForOneFace(target.frame, newRect);

	cv::Mat landmarkInputBlob = cv::dnn::blobFromImage(preprocessed, activescale, detailSize, detailMean, detailSwapRB, detailCrop);

	landmarkInputBlob.reshape(0, std::vector<int>{ 1, 1, preprocessed.rows, preprocessed.cols });

	detailNet.setInput(landmarkInputBlob, detailInputName);

	cv::Mat featureBlob = detailNet.forward(detailOutputName);
	size_t maxCols = std::min((featureBlob.cols / 2), TARGET_DETAIL_MODIFIER);
	for (int i = 0; i < maxCols; i++)
	{
		data[i].x = int(featureBlob.at<float>(0, 2 * i) * newRect.width + newRect.x);
		data[i].y = int(featureBlob.at<float>(0, 2 * i + 1) * newRect.height + newRect.y);
	}

}