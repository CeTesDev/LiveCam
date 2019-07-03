#include "detectors/DLibFaceDetector.h"

#include "CommonClasses.h"


#ifdef USE_VLAD_TRACKER


DLibFaceDetector::DLibFaceDetector()
{
	// We need a face detector.  We will use this to get bounding boxes for
	// each face in an image.
	detector = dlib::get_frontal_face_detector();

	// And we also need a shape_predictor.  This is the tool that will predict face
	// landmark positions given an image and face bounding box.  Here we are just
	// loading the model from the shape_predictor_68_face_landmarks.dat file
	dlib::deserialize("model/dlib/shape_predictor_68_face_landmarks.dat") >> shapePredictor;
}

cv::Mat DLibFaceDetector::preprocessFrame(cv::Mat frame)
{
	cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
	cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
	return gray;
}

std::vector<cv::Rect> DLibFaceDetector::FindObjects(cv::Mat frame)
{
	dlib::array2d<uchar> dlibImg;

	dlib::assign_image(dlibImg, dlib::cv_image<uchar>(frame));

	// Make the image larger so we can detect small faces.
	//pyramid_up(dlibImg);

	// Now tell the face detector to give us a list of bounding boxes
	// around all the faces in the image.
	std::vector<dlib::rectangle> rects = detector(dlibImg);

	std::vector<cv::Rect> faces(rects.size());

	for (int i = 0; i < rects.size(); ++i)
	{
		faces[i] = { (int)rects[i].left(), (int)rects[i].top(), (int)rects[i].width(), (int)rects[i].height() };
	}

	return faces;
}

void DLibFaceDetector::setScaleFactor(double newScaleFactor)
{
	activeScaleFactor = newScaleFactor;
}

#endif
