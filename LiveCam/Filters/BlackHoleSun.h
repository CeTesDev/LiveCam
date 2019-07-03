#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Graphics/GraphicsCommon.h>
#include <Graphics/GraphicsModel.h>
#include <FX.h>

class BlackHoleSun : public FX
{
public:
	BlackHoleSun();
	~BlackHoleSun();

	void transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *model) override;

private:
	
	std::array<std::array<ValueSmoother, TARGET_DETAIL_LIMIT * 2>, MAX_TO_TRACK> targetPointsSmoothers;
	
	void transformMeshOne(cv::Mat frame, const TrackingTarget& fxModel, Mesh2D& imgMesh);
};