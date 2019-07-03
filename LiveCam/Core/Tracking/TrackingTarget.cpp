#include "TrackingTarget.h"

#include <Common/Resolutions.h>

TrackingTarget::TrackingTarget()
{
	inited = false;
	pointId = 0;
	pointTotal = 0;

	frameWidth = 0;
	frameHeight = 0;

	pitch = 0;
	yaw = 0;
	roll = 0;

	width = 0;
	widthRaw = 0;

	xCenter = 0;
	xCenterRaw = 0;

	yCenter = 0;
	yCenterRaw = 0;

	pts.fill(0);
	confidence.fill(0);
	rect = { 0,0,0,0 };
}

TrackingTarget::TrackingTarget(const TrackingTarget& t)
{
	inited = t.inited;
	pointId = t.pointId;
	pointTotal = t.pointTotal;

	frameWidth = t.frameWidth;
	frameHeight = t.frameHeight;

	pitch = t.pitch;
	yaw = t.yaw;
	roll = t.roll;

	width = t.width;
	widthRaw = t.widthRaw;

	xCenter = t.xCenter;
	xCenterRaw = t.xCenterRaw;

	yCenter = t.yCenter;
	yCenterRaw = t.yCenterRaw;

	pts = t.pts;
	confidence = t.confidence;
	rect = t.rect;
	lastKnownRect = t.lastKnownRect;
	rotation = t.rotation;
}

TrackingTarget& TrackingTarget::operator=(const TrackingTarget& t)
{
	inited = t.inited;
	pointId = t.pointId;
	pointTotal = t.pointTotal;

	frameWidth = t.frameWidth;
	frameHeight = t.frameHeight;

	pitch = t.pitch;
	yaw = t.yaw;
	roll = t.roll;

	width = t.width;
	widthRaw = t.widthRaw;

	xCenter = t.xCenter;
	xCenterRaw = t.xCenterRaw;

	yCenter = t.yCenter;
	yCenterRaw = t.yCenterRaw;

	pts = t.pts;
	confidence = t.confidence;
	rect = t.rect;
	lastKnownRect = t.lastKnownRect;
	rotation = t.rotation;

	return *this;
}
