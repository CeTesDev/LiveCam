#include "FaceMapping.h"

namespace FaceMapping {

	const std::array<int, 6> LeftEyePoints =
	{
		37, 38, 39, 40, 41, 42
	};

	const std::array<int, 6> RightEyePoints =
	{
		43, 44, 45, 46, 47, 48
	};

	const std::array<int, 7> TopMouthPoints =
	{
		49, 50, 51, 52, 53, 54, 55
	};

	const std::array<int, 7> BottomMouthPoints =
	{
		61, 60, 59, 58, 57, 56, 65
	};

	Face::Face()
	{
	}

	Face::~Face()
	{
	}
	// ------------------------

	void Face::init(std::array<cv::Point2f, TARGET_DETAIL_LIMIT> pts)
	{
		top = pts[71];
		bottom = pts[8];
		for (int i = 0; i < TARGET_DETAIL_LIMIT; i++)
		{
			coords[i] = pts[i];
		}
	}

	std::vector<std::array<cv::Point2f, TOTAL_VERTEX_POINTS>> Face::getCoords()
	{
		std::vector<std::array<cv::Point2f, TOTAL_VERTEX_POINTS>> results;
		for (int i = 0; i < 2; i++)
		{
			std::array<cv::Point2f, TOTAL_VERTEX_POINTS> points;
			points.fill(cv::Point2f(0,0));
			int refdex = 0;
			for (auto &component : vertexIndex[i])
			{
				for (auto index : component)
				{
					points[refdex] = coords[index];
					refdex++;
				}
			}
			results.push_back(points);
		}
		
		return results;
	}

	void Face::updateCoords(ImagePoints points)
	{
		/*Eigen::Vector2f topPoint = { top.x, top.y };
		Eigen::Vector2f verticalOffset = points.getEigenAt(28) - Eigen::Vector2f(bottom.x, bottom.y);
		std::vector<std::array<Eigen::Vector2f, TOTAL_FOREHEAD_POINTS>> forehead =
		{
			{
				points.getEigenAt(1), points.getEigenAt(18) + verticalOffset * 0.5, points.getEigenAt(18),
				points.getEigenAt(19) + verticalOffset * 0.75, points.getEigenAt(19), points.getEigenAt(21) + verticalOffset * 0.95,
				points.getEigenAt(21), topPoint + verticalOffset, topPoint
			},
			{
				points.getEigenAt(17), points.getEigenAt(27) + verticalOffset * 0.5, points.getEigenAt(27),
				points.getEigenAt(26) + verticalOffset * 0.75, points.getEigenAt(26), points.getEigenAt(24) + verticalOffset * 0.95,
				points.getEigenAt(24), topPoint + verticalOffset, topPoint
			}
		};*/
		auto leftEyeCenter = PointCalcUtil::crossPointTwoLines(
			points.at(38),
			points.at(41),
			points.at(39),
			points.at(42)
		);

		auto rightEyeCenter = PointCalcUtil::crossPointTwoLines(
			points.at(44),
			points.at(47),
			points.at(45),
			points.at(48)
		);

		for (int i = 0; i < 6; ++i)
		{
			auto newLeftEye = leftEyeCenter + (points.at(LeftEyePoints[i]) - leftEyeCenter) * 1.4;
			points.pts[(LeftEyePoints[i] - 1) * 2] = newLeftEye.x;
			points.pts[(LeftEyePoints[i] - 1) * 2 + 1] = newLeftEye.y;

			auto newRightEye = rightEyeCenter + (points.at(RightEyePoints[i]) - rightEyeCenter) * 1.4;
			points.pts[(RightEyePoints[i] - 1) * 2] = newRightEye.x;
			points.pts[(RightEyePoints[i] - 1) * 2 + 1] = newRightEye.y;
		}

		/*auto mouthCenter = PointCalcUtil::crossPointTwoLines(
			points.at(53),
			points.at(59),
			points.at(54),
			points.at(60)
		);

		for (int i = 0; i < 7; ++i)
		{
			auto newTopMouth = mouthCenter + (points.at(TopMouthPoints[i]) - mouthCenter) * 1.4;
			points.pts[(TopMouthPoints[i] - 1) * 2] = newTopMouth.x;
			points.pts[(TopMouthPoints[i] - 1) * 2 + 1] = newTopMouth.y;

			auto newBottomMouth = mouthCenter + (points.at(BottomMouthPoints[i]) - mouthCenter) * 1.4;
			points.pts[(BottomMouthPoints[i] - 1) * 2] = newBottomMouth.x;
			points.pts[(BottomMouthPoints[i] - 1) * 2 + 1] = newBottomMouth.y;
		}*/

		for (int i = 0; i < TARGET_DETAIL_LIMIT; i++)
		{
			coords[i] = points.at(i + 1);
		}

		bottom = points.at(9);
		top = PointCalcUtil::crossPointTwoLines(bottom, points.at(28), points.at(21), points.at(24));
		int exLimit = EXTENDED_DETAIL_POINTS;
		float exWidthLeft = coords[29].x - coords[0].x;
		float exWidthRight = coords[16].x - coords[29].x;
		float exWidthOffset = round(exLimit / 2);
		float exHeight = coords[4].y - coords[29].y;
		float exStepXLeft = round(exWidthLeft / exWidthOffset);
		float exStepXRight = round(exWidthRight / exWidthOffset);
		float exStepY = exHeight;
		float exPointX = coords[0].x;
		float exPointY = round(coords[0].y - (exHeight / (exLimit - 1)));
		for (size_t j = TOTAL_DETAIL_POINTS; j < TARGET_DETAIL_LIMIT; ++j)
		{
			coords[j].x = exPointX;
			coords[j].y = exPointY;
			if (j < TARGET_TOP_INDEX)
			{
				exPointX += exStepXLeft;
				exStepY /= 2;
				exPointY -= exStepY;
			}
			else if (j > TARGET_TOP_INDEX)
			{
				exPointX += exStepXRight;
				exStepY *= 2;
				exPointY += exStepY;
			}
			else
			{
				exPointX = coords[29].x;
				exStepY /= 2;
				exPointY -= exStepY;
			}
		}
	}
}
