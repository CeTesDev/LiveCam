#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Tracking/_Constants.h>

class TrackingTarget {
public:

	Eigen::Matrix3d rotation;

	bool inited;

	int pointId;
	int pointTotal;

	int pitch;
	int yaw;
	int roll;

	int frameWidth;
	int frameHeight;

	float width;
	float widthRaw;

	float xCenter;
	float xCenterRaw;

	float yCenter;
	float yCenterRaw;

	std::array<float, TARGET_DETAIL_LIMIT * 2> pts;
	std::array<float, TARGET_DETAIL_LIMIT> confidence;

	cv::Rect rect;
	cv::Rect lastKnownRect;

public:

	TrackingTarget();
	TrackingTarget(const TrackingTarget& t);
	TrackingTarget& operator=(const TrackingTarget& t);

};

