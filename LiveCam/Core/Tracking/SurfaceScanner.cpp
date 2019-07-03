#include "SurfaceScanner.h"

#ifdef LC_INCLUDE_SURFACE_TRACKING

#include <Common/CommonClasses.h>

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

SurfaceScanner::Tracker::Tracker()
{
	initialized = false;
}

SurfaceScanner::Tracker::~Tracker()
{
}

std::vector<cv::Point2f> SurfaceScanner::Tracker::Points(std::vector<cv::KeyPoint> keypoints)
{
	std::vector<cv::Point2f> res;
	for (unsigned i = 0; i < keypoints.size(); i++) {
		res.push_back(keypoints[i].pt);
	}
	return res;
}

void SurfaceScanner::Tracker::init(std::shared_ptr<cv::Feature2D> _f2d, std::shared_ptr<cv::DescriptorMatcher> _matcher)
{
	f2d = _f2d;
	matcher = _matcher;
}

void SurfaceScanner::Tracker::setFirstFrame(const cv::Mat frame, std::vector<cv::Point2f> bb, Stats& stats)
{
	initialized = true;
	cv::Point *ptMask = new cv::Point[bb.size()];
	const cv::Point* ptContain = { &ptMask[0] };
	int iSize = static_cast<int>(bb.size());
	for (size_t i = 0; i < bb.size(); i++) {
		ptMask[i].x = static_cast<int>(bb[i].x);
		ptMask[i].y = static_cast<int>(bb[i].y);
	}
	first_frame = frame.clone();
	cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
	cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
	f2d->detectAndCompute(first_frame, matMask, first_kp, first_desc);
	stats.keypoints = (int)first_kp.size();
	object_bb = bb;
	delete[] ptMask;
}

cv::Rect SurfaceScanner::Tracker::process(const cv::Mat frame, Stats& stats)
{
	std::vector<cv::KeyPoint> kp;
	cv::Mat desc;

	f2d->detectAndCompute(frame, cv::noArray(), kp, desc);
	stats.keypoints = (int)kp.size();

	std::vector<std::vector<cv::DMatch>> matches;
	std::vector<cv::KeyPoint> matched1, matched2;
	matcher->knnMatch(first_desc, desc, matches, 2);
	for (unsigned i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			matched1.push_back(first_kp[matches[i][0].queryIdx]);
			matched2.push_back(kp[matches[i][0].trainIdx]);
		}
	}

	cv::Mat homography;
	if (matched1.size() >= 4) {
		homography = findHomography(Points(matched1), Points(matched2),
			cv::RANSAC, ransac_thresh);
	}

	if (matched1.size() < 4 || homography.empty()) {
		cv::Rect object(0, 0, frame.cols, frame.rows);
		return object;
	}

	std::vector<cv::Point2f> new_bb;
	perspectiveTransform(object_bb, new_bb, homography);

	int xLeftBottom = new_bb[3].x;
	int yLeftBottom = new_bb[3].y;
	int xRightTop = new_bb[1].x;
	int yRightTop = new_bb[1].y;

	cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
		(int)(xRightTop - xLeftBottom),
		(int)(yRightTop - yLeftBottom));

	object_bb = new_bb;

	return object;
}

SurfaceScanner::SurfaceScanner(bool useOrb)
{
	useOrbMethod = useOrb;
	originX = 0;
	originY = 0;
	std::shared_ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	if (useOrbMethod)
	{
		orb = cv::ORB::create();
		tracker.init(orb, matcher);
	}
	else
	{
		std::shared_ptr<cv::AKAZE> akaze = cv::AKAZE::create();
		tracker.init(akaze, matcher);
	}
	available = true;
}

SurfaceScanner::~SurfaceScanner()
{
	available = false;
}

void SurfaceScanner::start()
{}

void SurfaceScanner::stop()
{}

void SurfaceScanner::setRefFrame(cv::Mat frame, int x, int y)
{
	snapShot = frame.clone();

	originX = x;
	originY = y;
	//cv::resize(snapShot, snapShot, inputSize);

	std::vector<cv::Point2f> bb;
	cv::Rect uBox(0, 0, snapShot.cols, snapShot.rows);
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
	bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

	tracker.setFirstFrame(snapShot, bb, stats);
}

cv::Mat SurfaceScanner::prepScanFrame(cv::Mat frame)
{
	return frame;
}

cv::Mat SurfaceScanner::prepKeypointsFrame(cv::Mat frame)
{
	return frame;
}

std::vector<cv::Rect> SurfaceScanner::scan(cv::Mat frame)
{
	if (available)
	{
		available = false;
		std::lock_guard<std::mutex> frameLock(frameMutex);
		//cv::resize(frame, scanFrame, inputSize);
		scanFrame = frame.clone();
		std::thread t(&SurfaceScanner::asyncScan, this);
		t.detach();
	}

	return detectedSurfaces;
}

void SurfaceScanner::asyncScan()
{
	if (!scanFrame.empty() && tracker.initialized)
	{
		_surfaces.clear();
		if (useOrbMethod)
		{
			orb->setMaxFeatures(stats.keypoints);
		}
		cv::Rect rect = tracker.process(scanFrame, stats);
		_surfaces.push_back(rect);

		{
			std::lock_guard<std::mutex> lock2(scanMutex);
			this->detectedSurfaces = _surfaces;
			this->available = true;
		}
	}
}

void SurfaceScanner::keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data)
{
	std::vector<cv::Point2f> bb = tracker.getPoints();
	size_t maxIndex = std::min(TARGET_DETAIL_MODIFIER, (int)bb.size());
	for (size_t i = 0; i < maxIndex; i++) {
		data[i] = bb[i];
	}
}

void SurfaceScanner::scale(double newscale)
{
	activescale = newscale;
}

#endif
