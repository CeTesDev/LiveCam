#include "detectors/CvFaceDetector.h"

#include "CommonClasses.h"


#ifdef USE_VLAD_TRACKER



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
	// cv::setNumThreads(4);
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

	facemark->loadModel("model/caffe/lbfmodel.yaml");

	detectorAgregator->tracker->run();
}

CvFaceDetector::~CvFaceDetector()
{
	detectorAgregator->tracker->stop();
}

cv::Mat CvFaceDetector::preprocessFrame(cv::Mat frame)
{
	cv::Mat gray(frame.rows, frame.cols, CV_8UC3);
	cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
	return gray;
}
std::vector<cv::Rect> CvFaceDetector::FindObjects(cv::Mat frame)
{
	detectorAgregator->tracker->process(frame);

	std::vector<cv::Rect> faces;
	detectorAgregator->tracker->getObjects(faces);

	return faces;
}

cv::Mat CvFaceDetector::preprocessEntireFrame(cv::Mat frame)
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

void CvFaceDetector::FindLandmarks(FaceIntermediateStruct &face, std::array<cv::Point2f, LANDMARK_POINT_COUNT>& landmarks)
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

#endif
