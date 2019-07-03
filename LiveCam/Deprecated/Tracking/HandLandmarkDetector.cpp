#include "detectors/HandLandmarkDetector.h"


#include "CommonClasses.h"

#ifdef USE_VLAD_TRACKER

HandLandmarkDetector::HandLandmarkDetector()
{
	
}

cv::Mat HandLandmarkDetector::preprocessEntireFrame(cv::Mat frame)
{
	return frame;
}

cv::Mat HandLandmarkDetector::preprocessTargetFrame(cv::Mat frame, cv::Rect faceRect)
{
	return frame;
}

void HandLandmarkDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks)
{
	cv::Mat resizeFrame;
	cv::resize(face.faceFrame, resizeFrame, cv::Size(300, 300));
	// cv::Mat landmarkInputBlob = cv::dnn::blobFromImage(resizeFrame, 1 / 255.F, netInputSize, cv::Scalar(), false, false);

	// landmarkInputBlob.reshape(0, std::vector<int>{ 1, 1, netInputSize.width, netInputSize.height });

	// darkNet.setInput(landmarkInputBlob);

	cv::Mat detection;
	int H = detection.size[2];
	int W = detection.size[3];

	cv::Mat detectionMap(H, W, CV_32F, detection.ptr<float>());
	float confidenceThreshold = 0.3f;

	for (int i = 0; i < detectionMap.rows; i++)
	{
		float confidence = detectionMap.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			landmarks[i].x = static_cast<int>(detectionMap.at<float>(i, 3) * W);
			landmarks[i].y = static_cast<int>(detectionMap.at<float>(i, 4) * H);
		}
	}

}

#endif
