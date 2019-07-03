#include "BlackHoleSun.h"
#include <Widgets/FxWidgetFaceDistortion.h>
#include <Widgets/FxWidgetFacePinch.h>
#include <Graphics/GraphicsCommon.h>

BlackHoleSun::BlackHoleSun()
{
	auto targetdistortionShader = shaderManagerWrapper.LoadFromFile("./Assets/shaders/vertex/transparent.vertex", "./Assets/shaders/fragment/default.frag");

	auto targetDistortion = std::make_shared<GraphicsFaceDistortionModel>();
	targetDistortion->name = "BlackHoleSun";

	auto targetDistortionRenderParams = std::make_shared<ObjectRenderParams>();
	targetDistortionRenderParams->shader = targetdistortionShader;

	targetDistortion->objectRenderParams.fill({ targetDistortionRenderParams });

	targetDistortion->faceParams.fill({});

	models.push_back(targetDistortion);
}

BlackHoleSun::~BlackHoleSun()
{
}

void BlackHoleSun::transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *model)
{
	FxWidgetFacePinch pinchMaker(frame.size());
	Mesh2D srcMesh = pinchMaker.getSrcMesh();
	Mesh2D workMesh = pinchMaker.getDstMesh();

	for (int itarget = 0; itarget < targets.size(); itarget++)
	{
		transformMeshOne(frame, targets[itarget], workMesh);
	}

	cv::Size sz = workMesh.size();
	for (int iy = 0; iy < sz.height; iy++) {
		cv::Point2f pt = srcMesh.getPoint(0, iy);
		workMesh.setPoint(0, iy, pt.x, pt.y);
		pt = srcMesh.getPoint(sz.width - 1, iy);
		workMesh.setPoint(sz.width - 1, iy, pt.x, pt.y);
	}
	for (int ix = 0; ix < sz.width; ix++) {
		cv::Point2f pt = srcMesh.getPoint(ix, 0);
		workMesh.setPoint(ix, 0, pt.x, pt.y);
		pt = srcMesh.getPoint(ix, sz.height - 1);
		workMesh.setPoint(ix, sz.height - 1, pt.x, pt.y);
	}

	Mesh3DMaker maker;
	cv::Rect viewRegion(0, 0, frame.cols, frame.rows);
	maker.init(frame, viewRegion, srcMesh);
	maker.setDstMesh(workMesh);

	*model = maker.getMesh();
}

static cv::Point2f crossPointTwoLines(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4)
{
	float d = (p4.y - p3.y)*(p2.x - p1.x) - (p4.x - p3.x)*(p2.y - p1.y);
	if (d == 0)
		return cv::Point2f();

	float ua;
	ua = ((p4.x - p3.x)*(p1.y - p3.y) - (p4.y - p3.y)*(p1.x - p3.x)) / (float)d;
	cv::Point2f pt;
	pt.x = p1.x + ua*(p2.x - p1.x);
	pt.y = p1.y + ua*(p2.y - p1.y);
	return pt;
}

void BlackHoleSun::transformMeshOne(cv::Mat frame, const TrackingTarget& target, Mesh2D& imgMesh)
{
	cv::Size targetSize(target.frameWidth, target.frameHeight);
	FxWidgetFacePinch pinchMaker(targetSize, imgMesh);

	cv::Point2f leye = PointCalcUtil::crossPointTwoLines(
		cv::Point2f(target.pts[74], target.pts[75]),
		cv::Point2f(target.pts[80], target.pts[81]),
		cv::Point2f(target.pts[76], target.pts[77]),
		cv::Point2f(target.pts[72], target.pts[73])
	);
	cv::Point2f reye = PointCalcUtil::crossPointTwoLines(
		cv::Point2f(target.pts[86], target.pts[87]),
		cv::Point2f(target.pts[92], target.pts[93]),
		cv::Point2f(target.pts[88], target.pts[89]),
		cv::Point2f(target.pts[94], target.pts[95])
	);
	if (leye.x < 0 || leye.y < 0)
	{
		imgMesh = pinchMaker.getDstMesh();
		return;
	}

	std::array<float, TARGET_DETAIL_LIMIT * 2> smoothedPoints;

	for (int i = 0; i < TARGET_DETAIL_LIMIT; ++i)
	{
		smoothedPoints[i * 2] = targetPointsSmoothers[target.pointId][i * 2].smooth(target.pts[i * 2]);
		smoothedPoints[i * 2 + 1] = targetPointsSmoothers[target.pointId][i * 2 + 1].smooth(target.pts[i * 2 + 1]);
	}

	ImagePoints points(smoothedPoints);

	float leye_width = points.at(40).x - points.at(37).x;
	float reye_width = points.at(46).x - points.at(43).x;
	cv::Point2f center = PointCalcUtil::centerOf2Points(leye, reye);

	cv::Point2f chinpt = points.at(9);
	double rdist = PointCalcUtil::distanceBetween(chinpt, center);
	/*{
		float r = 0.55f;
		cv::Point2f d = leye - reye;
		cv::Point2f pt1, cross, vfrom, vto;

		pt1 = points.at(5);
		cross = crossPointTwoLines(pt1, pt1 + cv::Point2f(-d.y, d.x), leye, reye);
		vfrom = (cross - pt1*r)*(1.f / (1 - r));
		if (vfrom.y < 0) {
			imgMesh = pinchMaker.getDstMesh();
			return;
		}

		if (chinpt.y > frame.rows) {
			imgMesh = pinchMaker.getDstMesh();
			return;
		}
	}*/

	cv::Mat drawimage = frame.clone();
	for (int i = 0; i < 2; i++)
	{
		pinchMaker.doMagnifyWarpEllipse(leye, leye_width*1.5f, 0.3f, 0.4f);
	}

	for (int i = 0; i < 2; i++)
	{
		pinchMaker.doMagnifyWarpEllipse(reye, reye_width*1.5f, 0.3f, 0.4f);
	}
	

	center = points.at(34);
	const float extend_param = 1.0f;

	const float extend_start_parameter = 0.8f;
	for (int k = 0; k < 3; k++)
	{
		
		{
			float brush_size = rdist * 0.3f;
			float brush_pressure = 0.15f;
			cv::Point2f vto = PointCalcUtil::interPoint(points.at(4), center, extend_param);
			cv::Point2f vfrom = PointCalcUtil::interPoint(vto, center, extend_start_parameter);
			pinchMaker.doForwardWarp(vfrom, vto, brush_size, brush_pressure);
			cv::line(drawimage, vfrom, vto, cv::Scalar(0, 255, 0), 2);
			vto = PointCalcUtil::interPoint(points.at(14), center, extend_param);
			vfrom = PointCalcUtil::interPoint(vto, center, extend_start_parameter);
			pinchMaker.doForwardWarp(vfrom, vto, brush_size, brush_pressure);
			cv::line(drawimage, vfrom, vto, cv::Scalar(0, 255, 0), 2);
		}
		
	}
	cv::Point2f leftMouthCorner = points.at(49);
	cv::Point2f rightMouthCorner = points.at(55);
	cv::Point2f mouthCenter = PointCalcUtil::interPoint(leftMouthCorner, rightMouthCorner, 0.5f);

	const float mouth_size = PointCalcUtil::distanceBetween(leftMouthCorner, rightMouthCorner);
	
	
	pinchMaker.doMagnifyWarpEllipse(mouthCenter, mouth_size, 1.0f, 0.5f);

	const float arw[2] = { 1.0f, 0.5f };
	for (int k = 0; k < 2; k++) {
		for (int i = 32; i <= 36; i++)
		{
			if (i == 34)
				continue;
			cv::Point2f vfrom = points.at(i);
			cv::Point2f vto = points.at(30);
			if (i == 32) {
				cv::Point2f vdiff = points.at(32) - points.at(33);
				vfrom = vfrom + vdiff;
			}
			else if (i == 36)
			{
				cv::Point2f vdiff = points.at(36) - points.at(35);
				vfrom = vfrom + vdiff;
			}
			else {
				vfrom = PointCalcUtil::interPoint(vfrom, vto, 0.7f);
				continue;
			}
			vfrom = PointCalcUtil::interPoint(vfrom, vto, arw[k]);
			float nose_size = PointCalcUtil::distanceBetween(points.at(32), points.at(36));
			float brush_size = nose_size * 0.5f;
			pinchMaker.doForwardWarp(vfrom, vto, brush_size, 0.15f);
			cv::line(drawimage, vfrom, vto, cv::Scalar(0, 255, 0), 2);
		}
	}
	{
		for (int i = 32; i <= 36; i++)
		{
			if (i == 34)
				continue;
			cv::Point2f vfrom = points.at(i);
			cv::Point2f nose_center = (points.at(32) + points.at(36))*0.5f;
			if (i == 32) {
			}
			else if (i == 36)
			{
			}
			else {
				continue;
			}
			float nose_size = PointCalcUtil::distanceBetween(points.at(32), points.at(36));
			float brush_size = nose_size * 0.5f;
			pinchMaker.doForwardWarp(vfrom, nose_center, brush_size, 0.15f);
			cv::line(drawimage, vfrom, nose_center, cv::Scalar(0, 255, 0), 2);
		}
	}

	
	
	imgMesh = pinchMaker.getDstMesh();
}
