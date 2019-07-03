#include "detectors/DLibLandmarkDetector.h"

#include "CommonClasses.h"


#ifdef USE_VLAD_TRACKER


DLibLandmarkDetector::DLibLandmarkDetector()
{
	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	detector = dlib::get_frontal_face_detector();

	// And we also need a shape_predictor.  This is the tool that will predict face
	// landmark positions given an image and face bounding box.  Here we are just
	// loading the model from the shape_predictor_68_face_landmarks.dat file
	dlib::deserialize("model/dlib/shape_predictor_68_face_landmarks.dat") >> shapePredictor;
}

cv::Mat DLibLandmarkDetector::preprocessEntireFrame(cv::Mat frame)
{
	if (frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
		return gray;
	}
	else
	{
		return frame;
	}
}

void DLibLandmarkDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks)
{
	cv::Mat cv_gray = face.faceFrame;

	dlib::matrix<uchar> dlib_gray;
	dlib::assign_image(dlib_gray, dlib::cv_image<uchar>(cv_gray));

	//dlib::matrix<unsigned char> dlib_gray(face.faceFrame.rows, face.faceFrame.cols);

	//unsigned char* toBytePtr = (unsigned char*)dlib_gray.begin();
	//unsigned char* fromBytePtr = (unsigned char*)cv_gray.data;
	//std::copy(fromBytePtr, fromBytePtr + face.faceFrame.cols * face.faceFrame.rows, toBytePtr);

	dlib::rectangle rect(face.faceRect.x, face.faceRect.y, face.faceRect.x + face.faceRect.width, face.faceRect.y + face.faceRect.height);

	dlib::full_object_detection shape = shapePredictor(dlib_gray, rect);

	for (int i = 0; i < LANDMARK_POINT_COUNT; ++i)
	{
		auto point = shape.part(i);
		landmarks[i].x = point.x();
		landmarks[i].y = point.y();
	}
}

#endif
