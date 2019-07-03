#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>

#include <map>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <math.h>
#include <unordered_map>
#include <condition_variable>
#include <qdebug.h>
#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/algorithm/string.hpp>

#include <Common/PointCalcUtil.h>
#include <Tracking/ScannerInterface.h>
#include <Tracking/KeypointScannerInterface.h>

class TrackingTarget;

class Detector
{
public:
    Detector();
    ~Detector();

    void init(const std::string &targetTracker, const std::string &detailHandler, int lostTargetSleepMs, int lostTargetDelayMs, double scaleFactor);
    void start();
    void stop();
	void restart();
	void reset();
	void loadExtraTrackers();
	void scanTargets();
    void getTargetData(cv::Mat frame);
	void setSnapshot(cv::Mat frame, int x, int y);
    void setTargetLimit(int value);
	void setTargetRestriction(int value);
	void setTrackingType(std::string typeString);

	float scaleModifier = 0;

	cv::Mat scanImage;
	cv::Mat keypointImage;

	std::vector<TrackingTarget> getData(cv::Mat frame);

	std::string requestedTrackingType;
	TRACKER_TYPE activeTrackingType;

	std::vector<TrackingTarget> prevTargetData;
	std::vector<TrackingTarget> targetData;

    std::vector<TargetDetailStruct> getDetailData();

	std::shared_ptr<ScannerInterface> scanner = nullptr;
	std::map<TRACKER_TYPE, std::shared_ptr<ScannerInterface>> scanners;
	std::array<std::shared_ptr<KeypointScannerInterface>, MAX_TARGETS_COUNT> keypointScanners;

protected:

    bool started = false;
	bool minimalDetail = false;
	bool threadAvailable = true;
	bool isRestricted;

	int restrictedLimit;
	int unRestrictedLimit;
    int targetLimit;

	int currentRows;
	int currentCols;

    int lostTargetSleepMs;
    int lostTargetDelayMs;

    std::array<std::vector<std::pair<cv::Rect, cv::Mat>>, MAX_TARGETS_COUNT> targetsFound;

    cv::Mat syncLastFrame;
    int syncLastFrameIndex;

    std::array<TargetHoldingStruct, MAX_TARGETS_COUNT> _syncRects;
    std::array<TargetDetailStruct, MAX_TARGETS_COUNT> _syncDetails;
    std::array<TargetHoldingStruct, MAX_TARGETS_COUNT> _tmpRects;

    std::mutex targetLimitMutex;
    std::mutex frameMutex;
	std::mutex scannerMutex;
    std::mutex targetMutex;
    std::mutex landmarkMutex;

    bool syncStop = false;

    void sort(std::vector<cv::Rect> &targets, std::vector<cv::Mat> &frames, std::array<TargetHoldingStruct, MAX_TARGETS_COUNT> &targetRegistry);

};
