#include "Detector.h"

#include <Common/CommonClasses.h>

const cv::Size chessPatternSize(9, 6);
static cv::Mat cameraMatrix, distCoeffs;

static bool INTRINSICS_LOADED = false;

void loadIntrinsics()
{
	if (INTRINSICS_LOADED)
	{
		return;
	}
	INTRINSICS_LOADED = true;
	const std::string path = "Assets/model/surface/left_intrinsics.yml";
	cv::FileStorage intrinsicsFile(path, cv::FileStorage::READ);
	intrinsicsFile["camera_matrix"] >> cameraMatrix;
	intrinsicsFile["distortion_coefficients"] >> distCoeffs;
}

cv::Point2d fixPointPositionD(cv::Point2d A, cv::Point2d B, cv::Point2d P, float percent)
{
	cv::Point2d AP = P - A;
	cv::Point2d AB = B - A;
	double lenAB = sqrt(AB.x*AB.x + AB.y * AB.y);

	double lenAP = sqrt(AP.x*AP.x + AP.y * AP.y);

	double cosA = (AP.x *AB.x + AP.y * AB.y) / (lenAB * lenAP);

	double lenAK = lenAP * cosA;

	cv::Point2d K = A + AB * lenAK / lenAB;

	cv::Point2d KP = P - K;

	double lenKP = sqrt(KP.x*KP.x + KP.y * KP.y);

	if (lenKP > percent * lenAB)
	{
		double lenKPfixed = percent * lenAB;

		cv::Point2d Pfixed = K + lenKPfixed * KP / lenKP;

		return Pfixed;
	}

	return P;
}

namespace
{
	//! [compute-homography]
	cv::Mat computeHomography(const cv::Mat &R_1to2, const cv::Mat &tvec_1to2, const double d_inv, const cv::Mat &normal)
	{
		cv::Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
		return homography;
	}
	//! [compute-homography]

	cv::Mat computeHomography(const cv::Mat &R1, const cv::Mat &tvec1, const cv::Mat &R2, const cv::Mat &tvec2,
		const double d_inv, const cv::Mat &normal)
	{
		cv::Mat homography = R2 * R1.t() + d_inv * (-R2 * R1.t() * tvec1 + tvec2) * normal.t();
		return homography;
	}

	//! [compute-c2Mc1]
	void computeC2MC1(const cv::Mat &R1, const cv::Mat &tvec1, const cv::Mat &R2, const cv::Mat &tvec2,
		cv::Mat &R_1to2, cv::Mat &tvec_1to2)
	{
		//c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
		R_1to2 = R2 * R1.t();
		tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
	}
	//! [compute-c2Mc1]
}

TargetDetailStruct::TargetDetailStruct()
{
	modelPoints.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));
	modelPoints.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));
	modelPoints.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));
	modelPoints.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));
	modelPoints.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));
	modelPoints.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));
	modelPoints.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));
	modelPoints.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));
	modelPoints.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));
	modelPoints.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));
	modelPoints.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));
	modelPoints.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));
	modelPoints.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));
	modelPoints.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));

	poseMatrix = cv::Mat(3, 4, CV_64FC1);
	eulerAngles = cv::Mat(3, 1, CV_64FC1);

	boxModelSrc.push_back(cv::Point3d(-10, -10, 10));
	boxModelSrc.push_back(cv::Point3d(10, -10, 10));
	boxModelSrc.push_back(cv::Point3d(10, 10, 10));
	boxModelSrc.push_back(cv::Point3d(-10, 10, 10));
	boxModelSrc.push_back(cv::Point3d(-10, -10, -10));
	boxModelSrc.push_back(cv::Point3d(10, -10, -10));
	boxModelSrc.push_back(cv::Point3d(10, 10, -10));
	boxModelSrc.push_back(cv::Point3d(-10, 10, -10));

	pointId = 0;

	trackerType = TRACKER_TYPE::MISC;

	frameIndex = -1;
}

void TargetDetailStruct::facialProjection(size_t selfIndex)
{
	std::array<Eigen::Vector2f, TOTAL_DETAIL_POINTS + 2> predictions;
	std::array<Eigen::Vector2f, TOTAL_DETAIL_POINTS + 2> centeredPredictions;

	for (size_t i = 0; i < TOTAL_DETAIL_POINTS; i++) {
		predictions[i] = Eigen::Vector2f(
			data[i].x,
			data[i].y
		);
	}
	float minX = predictions[0](0);
	float maxX = predictions[0](0);
	float minY = predictions[0](1);
	float maxY = predictions[0](1);
	float sumX = predictions[0](0);
	float sumY = predictions[0](1);

	for (size_t i = 1; i < TOTAL_DETAIL_POINTS; i++)
	{
		if (minX > predictions[i](0))
		{
			minX = predictions[i](0);
		}

		if (minY > predictions[i](1))
		{
			minY = predictions[i](1);
		}

		if (maxX < predictions[i](0))
		{
			maxX = predictions[i](0);
		}

		if (maxY < predictions[i](1))
		{
			maxY = predictions[i](1);
		}

		sumX += predictions[i](0);
		sumY += predictions[i](1);
	}

	float avgX = sumX / static_cast<float>(TOTAL_DETAIL_POINTS);
	float avgY = sumY / static_cast<float>(TOTAL_DETAIL_POINTS);
	Eigen::Vector2f avgXY = Eigen::Vector2f(avgX, avgY);
	Eigen::Vector2f avgWH = Eigen::Vector2f(Resolutions::INPUT_ACTUAL_WIDTH / 2, Resolutions::INPUT_ACTUAL_HEIGHT / 2);

	for (size_t i = 0; i < TOTAL_DETAIL_POINTS; i++)
	{
		centeredPredictions[i] = predictions[i] - Eigen::Vector2f(avgX, avgY) + Eigen::Vector2f(Resolutions::INPUT_ACTUAL_WIDTH / 2, Resolutions::INPUT_ACTUAL_HEIGHT / 2);
	}

	predictions = centeredPredictions;

	cv::Point2d mouthLeftCorner = cv::Point2d(predictions[48](0), predictions[48](1));
	cv::Point2d mouthRightCorner = cv::Point2d(predictions[54](0), predictions[54](1));
	cv::Point2d mouthCentralBottomCorner = cv::Point2d(predictions[57](0), predictions[57](1));

	cv::Point2d mouthCentralBottomCornerFixed;
	if (mouthLeftCorner.x == 0 && mouthLeftCorner.y == 0 && mouthRightCorner.x == 0 && mouthRightCorner.y == 0)
	{
		mouthCentralBottomCornerFixed = { 0.0, 0.0 };
	}
	else
	{
		mouthCentralBottomCornerFixed = fixPointPositionD(mouthLeftCorner, mouthRightCorner, mouthCentralBottomCorner, 0.05);
	}

	cv::Point2d ChinLeftPoint = cv::Point2d(predictions[0](0), predictions[0](1));
	cv::Point2d ChinRightPoint = cv::Point2d(predictions[16](0), predictions[16](1));
	cv::Point2d ChinCenterPoint = cv::Point2d(predictions[8](0), predictions[8](1));

	cv::Point2d ChinCenterPointFixed;

	if (ChinLeftPoint.x == 0 && ChinLeftPoint.y == 0 && ChinRightPoint.x == 0 && ChinRightPoint.y == 0)
	{
		ChinCenterPointFixed = { 0.0, 0.0 };
	}
	else
	{
		ChinCenterPointFixed = fixPointPositionD(ChinLeftPoint, ChinRightPoint, ChinCenterPoint, 0.7);

	}

	std::vector<cv::Point2d> imagePoints;
	imagePoints.push_back(cv::Point2d(predictions[17](0), predictions[17](1)));
	imagePoints.push_back(cv::Point2d(predictions[21](0), predictions[21](1)));
	imagePoints.push_back(cv::Point2d(predictions[22](0), predictions[22](1)));
	imagePoints.push_back(cv::Point2d(predictions[26](0), predictions[26](1)));
	imagePoints.push_back(cv::Point2d(predictions[36](0), predictions[36](1)));
	imagePoints.push_back(cv::Point2d(predictions[39](0), predictions[39](1)));
	imagePoints.push_back(cv::Point2d(predictions[42](0), predictions[42](1)));
	imagePoints.push_back(cv::Point2d(predictions[45](0), predictions[45](1)));
	imagePoints.push_back(cv::Point2d(predictions[31](0), predictions[31](1)));
	imagePoints.push_back(cv::Point2d(predictions[35](0), predictions[35](1)));
	imagePoints.push_back(mouthLeftCorner);
	imagePoints.push_back(mouthRightCorner);
	imagePoints.push_back(mouthCentralBottomCornerFixed);
	imagePoints.push_back(ChinCenterPointFixed);
	
	cv::Point2d center = cv::Point2d(Resolutions::INPUT_ACTUAL_WIDTH / 2, Resolutions::INPUT_ACTUAL_HEIGHT / 2);
	
	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << Resolutions::INPUT_ACTUAL_WIDTH, 0, center.x, 0, Resolutions::INPUT_ACTUAL_WIDTH, center.y, 0, 0, 1);

	cv::Mat distortionCoefficients = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
	cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distortionCoefficients, rotationVector, translationVector);
	cv::projectPoints(boxModelSrc, rotationVector, translationVector, cameraMatrix, distortionCoefficients, boxModelDst);
	cv::Rodrigues(rotationVector, rotationMatrix);
	cv::hconcat(rotationMatrix, translationVector, poseMatrix);

	cv::Mat outIntrinsics = cv::Mat(3, 3, CV_64FC1);
	cv::Mat outRotation = cv::Mat(3, 3, CV_64FC1);
	cv::Mat outTranslation = cv::Mat(3, 1, CV_64FC1);
	cv::decomposeProjectionMatrix(poseMatrix, outIntrinsics, outRotation, outTranslation, cv::noArray(), cv::noArray(), cv::noArray(), eulerAngles);

	frameIndex = -1;
}

void TargetDetailStruct::handProjection(size_t selfIndex)
{

	frameIndex = -1;
}

void TargetDetailStruct::surfaceProjection(size_t selfIndex)
{
	loadIntrinsics();

	cv::Point2d center = cv::Point2d(Resolutions::INPUT_ACTUAL_WIDTH / 2, Resolutions::INPUT_ACTUAL_HEIGHT / 2);
	cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) << Resolutions::INPUT_ACTUAL_WIDTH, 0, center.x, 0, Resolutions::INPUT_ACTUAL_WIDTH, center.y, 0, 0, 1);

	std::vector<cv::Point2d> imagePoints;
	for (size_t i = 0; i < TOTAL_DETAIL_POINTS; i++) {
		imagePoints.push_back(cv::Point2d(data[i].x, data[i].y));
	}

	//! [compute-poses]
	std::vector<cv::Point2f> corners1, corners2;
	if (!cv::findChessboardCorners(frame, chessPatternSize, corners1))
	{
		std::cout << "Error, cannot find the chessboard corners in both images." << endl;
		return;
	}
	if (!cv::findChessboardCorners(lastKnownFrame, chessPatternSize, corners2))
	{
		std::cout << "Error, cannot find the chessboard corners in both images." << endl;
		return;
	}

	std::vector<cv::Point3f> objectPoints;
	objectPoints.resize(0);

	cv::Mat rvec1, tvec1;
	cv::solvePnP(objectPoints, corners1, cameraMatrix, distCoeffs, rvec1, tvec1);
	cv::Mat rvec2, tvec2;
	cv::solvePnP(objectPoints, corners2, cameraMatrix, distCoeffs, rvec2, tvec2);
	//! [compute-poses]

	//! [compute-camera-displacement]
	cv::Mat R1, R2;
	cv::Rodrigues(rvec1, R1);
	cv::Rodrigues(rvec2, R2);

	cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);
	cv::projectPoints(boxModelSrc, rotationVector, translationVector, cameraMatrix, distCoeffs, boxModelDst);

	cv::Mat R_1to2, t_1to2;
	computeC2MC1(R1, tvec1, R2, tvec2, rotationVector, translationVector);
	cv::Rodrigues(rotationVector, rotationMatrix);
	cv::hconcat(rotationMatrix, translationVector, poseMatrix);
	//! [compute-camera-displacement]

	cv::Mat outIntrinsics = cv::Mat(3, 3, CV_64FC1);
	cv::Mat outRotation = cv::Mat(3, 3, CV_64FC1);
	cv::Mat outTranslation = cv::Mat(3, 1, CV_64FC1);
	cv::decomposeProjectionMatrix(poseMatrix, outIntrinsics, outRotation, outTranslation, cv::noArray(), cv::noArray(), cv::noArray(), eulerAngles);

	frameIndex = -1;
}

void TargetDetailStruct::getProjection(size_t selfIndex)
{
	switch (trackerType)
	{
		case TRACKER_TYPE::FACE:
			facialProjection(selfIndex);
			break;
		case TRACKER_TYPE::HANDS:
			handProjection(selfIndex);
			break;
		case TRACKER_TYPE::SURFACE:
			surfaceProjection(selfIndex);
	}
}
