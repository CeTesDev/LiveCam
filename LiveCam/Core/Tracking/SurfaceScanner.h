#pragma once

#ifdef LC_INCLUDE_SURFACE_TRACKING

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Tracking/ScannerInterface.h>


class SurfaceScanner : public ScannerInterface
{
	struct Stats
	{
		int matches;
		int inliers;
		double ratio;
		int keypoints;
		double fps;

		Stats() : matches(0),
			inliers(0),
			ratio(0),
			keypoints(0),
			fps(0.)
		{}

		Stats& operator+=(const Stats& op) {
			matches += op.matches;
			inliers += op.inliers;
			ratio += op.ratio;
			keypoints += op.keypoints;
			fps += op.fps;
			return *this;
		}
		Stats& operator/=(int num)
		{
			matches /= num;
			inliers /= num;
			ratio /= num;
			keypoints /= num;
			fps /= num;
			return *this;
		}
	};

	class Tracker
	{
	public:
		Tracker();
		~Tracker();

		void init(std::shared_ptr<cv::Feature2D> _f2d, std::shared_ptr<cv::DescriptorMatcher> _matcher);
		void setFirstFrame(const cv::Mat frame, std::vector<cv::Point2f> bb, Stats& stats);
		cv::Rect process(const cv::Mat frame, Stats& stats);
		std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints);
		std::vector<cv::Point2f> getPoints()
		{
			return object_bb;
		};
		std::shared_ptr<cv::Feature2D> getDetector() {
			return f2d;
		}

		bool initialized = false;

	protected:
		std::shared_ptr<cv::Feature2D> f2d;
		std::shared_ptr<cv::DescriptorMatcher> matcher;
		cv::Mat first_frame, first_desc;
		std::vector<cv::KeyPoint> first_kp;
		std::vector<cv::Point2f> object_bb;
	};

public:
	SurfaceScanner(bool useOrb);
	~SurfaceScanner();

	Tracker tracker;
	bool available = false;

	Stats stats;

	int originX, originY;

	cv::Mat snapShot;
	cv::Mat scanFrame;
	std::vector<cv::Rect> _surfaces;
	std::vector<cv::Rect> detectedSurfaces;
	std::vector<cv::Point2f> _points;
	std::vector<cv::Point2f> detectedPoints;

	std::shared_ptr<cv::ORB> orb;

	bool useOrbMethod = false;

	void asyncScan();
	void start() override;
	void stop() override;
	void setRefFrame(cv::Mat frame, int x, int y) override;
	cv::Mat prepScanFrame(cv::Mat frame) override;
	cv::Mat prepKeypointsFrame(cv::Mat frame) override;
	std::vector<cv::Rect> scan(cv::Mat frame) override;
	void keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data) override;
	void scale(double newscale) override;

protected:
	cv::Size inputSize = cv::Size(300, 300);
	std::mutex frameMutex;
	std::mutex scanMutex;

};

#endif
