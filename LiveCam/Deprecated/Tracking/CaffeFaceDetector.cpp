#include "detectors/CaffeFaceDetector.h"

#include "CommonClasses.h"


#ifdef USE_VLAD_TRACKER

CaffeFaceDetector::CaffeFaceDetector()
{
	const cv::String faceTrackerConfiguration = "model/caffe/deploy.prototxt";
	const cv::String faceTrackerBinary = "model/caffe/res10_300x300_ssd_iter_140000.caffemodel";

	faceNet = cv::dnn::readNetFromCaffe(faceTrackerConfiguration, faceTrackerBinary);
}

cv::Mat CaffeFaceDetector::preprocessFrame(cv::Mat frame)
{
	return frame;
}

std::vector<cv::Rect> CaffeFaceDetector::FindObjects(cv::Mat frame)
{
	//if (frame.channels() == 4)
	//{
	//	cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
	//}


	// Resize blob image to 300 X 300
	cv::Mat resizeFrame;
	cv::resize(frame, resizeFrame, cv::Size(300, 300));
	cv::Mat inputBlob = cv::dnn::blobFromImage(resizeFrame, faceTrackerScaleFactor, cv::Size(faceTrackerWidth, faceTrackerHeight), faceTrackerMeanVal, false, false);

	faceNet.setInput(inputBlob, "data");

	cv::Mat detection = faceNet.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//float confidenceThreshold = 0.5f;
	float confidenceThreshold = 0.3f;

	std::vector<cv::Rect> faces;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

			cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));


			/*
			if (object.width > object.height)
			{
				int diff = object.width - object.height;
				object.y -= diff / 2;
				object.height += diff;
			}
			else if (object.width < object.height)
			{
				int diff = object.height - object.width;
				object.x -= diff / 2;
				object.width += diff;
			}*/

			faces.push_back(object);
		}
	}

	return faces;

}

void CaffeFaceDetector::setScaleFactor(double newScaleFactor)
{
	activeScaleFactor = newScaleFactor;
}

#endif
