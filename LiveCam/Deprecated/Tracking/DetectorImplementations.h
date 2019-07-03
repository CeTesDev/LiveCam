#pragma once

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>

#include <detectors/DetectorTypes.h>

#include "ValueSmoother.h"



class FaceDetector
{
public:
	virtual cv::Mat preprocessFrame(cv::Mat frame) = 0;
	virtual std::vector<cv::Rect> FindObjects(cv::Mat frame) = 0;
};



class FaceLandmarkDetector
{
public:
	virtual cv::Mat preprocessFrameForAllFaces(cv::Mat frame) = 0; // for all faces
	virtual cv::Mat preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect) = 0; // for one face
	virtual void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks) = 0;
};


class CaffeFaceDetector : public FaceDetector
{
public:
	CaffeFaceDetector();

	cv::Mat preprocessFrame(cv::Mat frame) override;
	std::vector<cv::Rect> FindObjects(cv::Mat frame) override;

protected:
	cv::dnn::Net faceNet;
};


class CaffeLandmarkDetector : public FaceLandmarkDetector
{
public:
	CaffeLandmarkDetector(bool use3D);

	cv::Mat preprocessFrameForAllFaces(cv::Mat frame) override; // for all faces
	cv::Mat preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect) override; // for one face
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks) override;

protected:

	bool use3D;
	cv::Size netInputSize;
	cv::String netString;
	cv::dnn::Net landmarkNet;
};



class DLibFaceDetector : public FaceDetector
{
public:
	DLibFaceDetector();

	cv::Mat preprocessFrame(cv::Mat frame) override;
	std::vector<cv::Rect> FindObjects(cv::Mat frame) override;

private:
	dlib::frontal_face_detector detector;
	dlib::shape_predictor shapePredictor;
};




class DLibLandmarkDetector : public FaceLandmarkDetector
{
public:
	DLibLandmarkDetector();

	cv::Mat preprocessFrameForAllFaces(cv::Mat frame) override; // for all faces
	cv::Mat preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect) override; // for one face
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks) override;

private:
	dlib::frontal_face_detector detector;
	dlib::shape_predictor shapePredictor;
};



class CvFaceDetector : public FaceLandmarkDetector, public FaceDetector
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

	cv::Mat preprocessFrame(cv::Mat frame) override; // FaceDetector
	std::vector<cv::Rect> FindObjects(cv::Mat frame) override;
	cv::Mat preprocessFrameForAllFaces(cv::Mat frame) override; // for all faces, FaceLandmarkDetector
	cv::Mat preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect) override; // for one face, FaceLandmarkDetector
	void setScaleFactor(double newScaleFactor) override;
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks) override;

private:
	cv::Ptr<DetectorAgregator> detectorAgregator;
	cv::Ptr<cv::face::FacemarkLBF> facemark;
};
