#include "detectors/DetectorImplementations.h"

#include "CommonClasses.h"


static const size_t faceTrackerWidth = 300;
static const size_t faceTrackerHeight = 300;

static const double faceTrackerScaleFactor = 1.0;
static const cv::Scalar faceTrackerMeanVal(104.0, 177.0, 123.0);




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

	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, faceTrackerScaleFactor, cv::Size(faceTrackerWidth, faceTrackerHeight), faceTrackerMeanVal, false, false);

	faceNet.setInput(inputBlob, "data");

	cv::Mat detection = faceNet.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//float confidenceThreshold = 0.5f;
	float confidenceThreshold = 0.2f;

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
			}


			/*
			object.x = object.x - object.width*0.05f;
			object.y = object.y - object.height*0.05f;
			object.width = object.width*1.1f;
			object.height = object.height*1.1f;
			*/
			faces.push_back(object);
		}
	}

	return faces;

}



CaffeLandmarkDetector::CaffeLandmarkDetector(bool use3D)
{
	this->use3D = use3D;

	const cv::String configuration3D = "model/caffe/fa_deploy.prototxt";
	const cv::String binary3D = "model/caffe/fa__iter_1400000.caffemodel";

	const cv::String configuration2D = "model/caffe/landmark_deploy.prototxt";
	const cv::String binary2D = "model/caffe/VanFace.caffemodel";

	landmarkNet = use3D ?
		cv::dnn::readNetFromCaffe(configuration3D, binary3D) :
		cv::dnn::readNetFromCaffe(configuration2D, binary2D);

	netInputSize = use3D ? cv::Size(40, 40) : cv::Size(60, 60);

	netString = use3D ? "Dense2" : "Dense3"; // ?????????????????????/
}

cv::Mat CaffeLandmarkDetector::preprocessFrameForAllFaces(cv::Mat frame)
{
	if (frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, CV_RGB2GRAY);
		return gray;
	}
	else
	{
		return frame;
	}
}

cv::Mat CaffeLandmarkDetector::preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect)
{
	cv::Mat srcROI(frame, faceRect);

	cv::Mat gray;
	srcROI.convertTo(gray, CV_32FC1);

	cv::Mat smallGray = cv::Mat(netInputSize, CV_32FC1);
	cv::resize(gray, smallGray, netInputSize, 0, 0, cv::INTER_CUBIC);

	cv::Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	meanStdDev(smallGray, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);

	smallGray = (smallGray - m) / (0.000001 + sd);

	return smallGray;
}

void CaffeLandmarkDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks)
{
	cv::Mat preprocessed = preprocessFrameForOneFace(face.faceFrame, face.faceRect);

	cv::Mat landmarkInputBlob = cv::dnn::blobFromImage(preprocessed, 1.0, netInputSize, cv::Scalar(), false, false);

	landmarkInputBlob.reshape(0, std::vector<int>{ 1, 1, preprocessed.rows, preprocessed.cols });

	landmarkNet.setInput(landmarkInputBlob);

	cv::Mat featureBlob = landmarkNet.forward(netString);

	for (int i = 0; i < featureBlob.cols / 2; i++)
	{
		landmarks[i].x = int(featureBlob.at<float>(0, 2 * i) * face.faceRect.width + face.faceRect.x);
		landmarks[i].y = int(featureBlob.at<float>(0, 2 * i + 1) * face.faceRect.height + face.faceRect.y);
	}
}




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
	cvtColor(frame, gray, CV_RGB2GRAY);
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

cv::Mat DLibLandmarkDetector::preprocessFrameForAllFaces(cv::Mat frame)
{
	if (frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, CV_RGB2GRAY);
		return gray;
	}
	else
	{
		return frame;
	}
}

cv::Mat DLibLandmarkDetector::preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect)
{
	return frame;
}

void DLibLandmarkDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks)
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

	for (int i = 0; i < 68; ++i)
	{
		auto point = shape.part(i);
		landmarks[i].x = point.x();
		landmarks[i].y = point.y();
	}
}




CvFaceDetector::CascadeDetectorAdapter::CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
{
}

CvFaceDetector::CascadeDetectorAdapter::~CascadeDetectorAdapter()
{
}

void CvFaceDetector::CascadeDetectorAdapter::detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
{
	Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
}

CvFaceDetector::DetectorAgregator::DetectorAgregator(cv::Ptr<CascadeDetectorAdapter> _mainDetector, cv::Ptr<CascadeDetectorAdapter> _trackingDetector) :
	mainDetector(_mainDetector),
	trackingDetector(_trackingDetector)
{
	cv::DetectionBasedTracker::Parameters DetectorParams;
	tracker = cv::makePtr<cv::DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
}

CvFaceDetector::CvFaceDetector()
{
	cv::FileStorage fs("model/cvFace/lbpcascade_frontalface.xml", cv::FileStorage::READ);

	cv::Ptr<cv::CascadeClassifier> mainCascadeClassifier = cv::makePtr<cv::CascadeClassifier>();
	cv::Ptr<cv::CascadeClassifier> trackingCascadeClassifier = cv::makePtr<cv::CascadeClassifier>();

	mainCascadeClassifier->read(fs.getFirstTopLevelNode());
	trackingCascadeClassifier->read(fs.getFirstTopLevelNode());

	cv::Ptr<CascadeDetectorAdapter> mainDetector = cv::makePtr<CascadeDetectorAdapter>(mainCascadeClassifier);
	cv::Ptr<CascadeDetectorAdapter> trackingDetector = cv::makePtr<CascadeDetectorAdapter>(trackingCascadeClassifier);

	double scaledSize = 150 * activeScaleFactor;

	mainDetector->setMinObjectSize(cv::Size(scaledSize, scaledSize));

	detectorAgregator = cv::makePtr<DetectorAgregator>(mainDetector, trackingDetector);

	facemark = cv::face::FacemarkLBF::create();

	facemark->loadModel("model/cvFace/lbfmodel.yaml");

	detectorAgregator->tracker->run();
}

CvFaceDetector::~CvFaceDetector()
{
	detectorAgregator->tracker->stop();
}

cv::Mat CvFaceDetector::preprocessFrame(cv::Mat frame)
{
	cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
	cvtColor(frame, gray, CV_RGB2GRAY);
	return gray;
}
std::vector<cv::Rect> CvFaceDetector::FindObjects(cv::Mat frame)
{
	detectorAgregator->tracker->process(frame);

	std::vector<cv::Rect> faces;
	detectorAgregator->tracker->getObjects(faces);

	return faces;
}

cv::Mat CvFaceDetector::preprocessFrameForOneFace(cv::Mat frame, cv::Rect faceRect)
{
	return frame;
}

cv::Mat CvFaceDetector::preprocessFrameForAllFaces(cv::Mat frame)
{
	if (frame.channels() == 3)
	{
		cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
		cvtColor(frame, gray, CV_RGB2GRAY);
		return gray;
	}
	else
	{
		return frame;
	}
}

void CvFaceDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, 68>& landmarks)
{
	std::vector<cv::Rect> ROI = { face.faceRect };
	std::vector<std::vector<cv::Point2f>> shapes;

	facemark->fit(face.faceFrame, ROI, shapes);

	std::copy(shapes[0].begin(), shapes[0].end(), landmarks.begin());
}

void CvFaceDetector::setScaleFactor(double newScaleFactor)
{
	activeScaleFactor = newScaleFactor;
}
