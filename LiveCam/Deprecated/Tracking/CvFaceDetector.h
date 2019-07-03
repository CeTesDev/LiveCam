#pragma once

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>

#include <detectors/DetectorTypes.h>

#include "ValueSmoother.h"

#include "ObjectDetectorInterface.h"
#include "LandmarkDetectorInterface.h"

#ifdef USE_VLAD_TRACKER

class CvFaceDetector : public LandmarkDetectorInterface, public ObjectDetectorInterface
{
public:

	class CascadeDetectorAdapter : public cv::DetectionBasedTracker::IDetector
	{
	public:
		CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector);
		virtual ~CascadeDetectorAdapter();

		void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects);

	private:
		CascadeDetectorAdapter();
		cv::Ptr<cv::CascadeClassifier> Detector;
	};

	struct DetectorAgregator
	{
		cv::Ptr<CascadeDetectorAdapter> mainDetector;
		cv::Ptr<CascadeDetectorAdapter> trackingDetector;

		cv::Ptr<cv::DetectionBasedTracker> tracker;
		DetectorAgregator(cv::Ptr<CascadeDetectorAdapter> _mainDetector, cv::Ptr<CascadeDetectorAdapter> _trackingDetector);
	};

	CvFaceDetector();
	~CvFaceDetector();

	double activeScaleFactor = 1.0;

	cv::Mat preprocessFrame(cv::Mat frame) override; // FaceDetector
	std::vector<cv::Rect> FindObjects(cv::Mat frame) override;
	cv::Mat preprocessEntireFrame(cv::Mat frame) override; // for all faces, FaceLandmarkDetector
	void setScaleFactor(double newScaleFactor) override;
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks) override;

private:
	cv::Ptr<DetectorAgregator> detectorAgregator;
	cv::Ptr<cv::face::FacemarkLBF> facemark;
};

#endif
