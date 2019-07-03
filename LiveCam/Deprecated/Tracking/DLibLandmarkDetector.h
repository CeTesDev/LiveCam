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


class DLibLandmarkDetector : public LandmarkDetectorInterface
{
public:
	DLibLandmarkDetector();

	cv::Mat preprocessEntireFrame(cv::Mat frame) override; // for all faces
	void FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks) override;

private:
	dlib::frontal_face_detector detector;
	dlib::shape_predictor shapePredictor;
};


#endif
