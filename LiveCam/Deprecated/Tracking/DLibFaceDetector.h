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

class DLibFaceDetector : public ObjectDetectorInterface
{
public:
	DLibFaceDetector();

	double activeScaleFactor = 1.0;

	cv::Mat preprocessFrame(cv::Mat frame) override;
	std::vector<cv::Rect> FindObjects(cv::Mat frame) override;
	void setScaleFactor(double newScaleFactor) override;

private:
	dlib::frontal_face_detector detector;
	dlib::shape_predictor shapePredictor;
};


#endif
