#pragma once

#include <opencv2/core.hpp>
#include <Common/Resolutions.h>
#include <Tracking/_Constants.h>

constexpr size_t trackerWidth = 300;
constexpr size_t trackerHeight = 300;

struct TrackerConfigStruct
{
	bool swapRB = false;
	bool crop = false;

	cv::Size size;
	cv::Scalar mean = cv::Scalar(0);
	cv::String inputName;
	cv::String outputName;
	std::vector<cv::String> outputNames;
};

struct TrackingTargetStruct
{
    int templateIndex;
    cv::Rect rect;
    cv::Mat frame;
    int modelID;
    int targetIndex;
};

struct TrackingTargetRectStruct
{
    bool result;
    double x_abs;
    double y_abs;
    double w_abs;
    double h_abs;
};

struct TargetPositionStruct
{
	cv::Rect rect;
	int state = 0;
	int index = 0;
	float xCenter = 0.0;
	float yCenter = 0.0;
};

struct TargetHoldingStruct
{
    bool visible = false;

    int pointId;

    cv::Mat frame;
	cv::Mat lastKnownFrame;
	cv::Rect rect;
    cv::Rect lastKnownRect;

    std::chrono::time_point<std::chrono::system_clock> lastKnownTime;

    int frameIndex;
};

struct TargetDetailStruct
{
    bool visible = false;
	bool minimalDetail = false;

	TRACKER_TYPE trackerType;

	cv::Mat frame;
	cv::Mat lastKnownFrame;
	cv::Rect rect;
    cv::Rect lastKnownRect;

    std::array<cv::Point2f, TARGET_DETAIL_MODIFIER> data;

    int frameIndex;

    int pointId;

    std::vector<cv::Point3f> modelPoints;
    cv::Mat rotationVector;
    cv::Mat translationVector;
    cv::Mat rotationMatrix;
    cv::Mat poseMatrix;
    cv::Vec3d eulerAngles;

    std::vector<cv::Point3d> boxModelSrc;
    std::vector<cv::Point2d> boxModelDst;

    TargetDetailStruct();

	void getProjection(size_t selfIndex);
    void facialProjection(size_t selfIndex);
	void handProjection(size_t selfIndex);
	void surfaceProjection(size_t selfIndex);
};

