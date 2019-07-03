#pragma once

#include <vector>
#include <array>
#include <opencv2/core.hpp>
#include <Common/CommonClasses.h>
#include <Common/PointCalcUtil.h>

namespace FaceMapping
{
	static const size_t LINE_STRIP_COUNT = 16;

	static const size_t CHIN_POINTS_START = 0;
	static const size_t TOTAL_CHIN_POINTS = 9;

	static const size_t CHEEK_POINTS_START = 9;
	static const size_t TOTAL_CHEEK_POINTS = 10;

	static const size_t BROW_POINTS_START = 19;
	static const size_t TOTAL_BROW_POINTS = 10;

	static const size_t NOSE_POINTS_START = 29;
	static const size_t TOTAL_NOSE_POINTS = 8;

	static const size_t PHILTRUM_POINTS_START = 37;
	static const size_t TOTAL_PHILTRUM_POINTS = 6;

	static const size_t FOREHEAD_POINTS_START = 43;
	static const size_t TOTAL_FOREHEAD_POINTS = 10;

	static const size_t MOUTH_POINTS_START = 53;
	static const size_t TOTAL_MOUTH_POINTS = 7;

	static const size_t EYE_POINTS_START = 60;
	static const size_t TOTAL_EYE_POINTS = 6;

	static const size_t TOTAL_VERTEX_POINTS = TOTAL_CHIN_POINTS + 
		TOTAL_CHEEK_POINTS + TOTAL_BROW_POINTS + 
		TOTAL_NOSE_POINTS + TOTAL_PHILTRUM_POINTS + 
		TOTAL_EYE_POINTS + TOTAL_MOUTH_POINTS + 
		TOTAL_FOREHEAD_POINTS;

	static const cv::Point2f TopPoint = { 0.5f, 0.2f };
	static const cv::Point2f TexMapTopPoint = { 0.503125f, 0.279688f };

	static const std::array<cv::Point2f, TARGET_DETAIL_LIMIT> InitTable = {
		cv::Point2f{ 100, 348 }, // 0
		cv::Point2f{ 110, 439 }, // 1
		cv::Point2f{ 131, 543 }, // 2
		cv::Point2f{ 142, 642 }, // 3
		cv::Point2f{ 196, 738 }, // 4
		cv::Point2f{ 243, 822 }, // 5
		cv::Point2f{ 312, 890 }, // 6
		cv::Point2f{ 401, 948 }, // 7
		cv::Point2f{ 500, 954 }, // 8
		cv::Point2f{ 603, 946 }, // 9
		cv::Point2f{ 688, 890 }, // 10
		cv::Point2f{ 758, 822 }, // 11
		cv::Point2f{ 814, 738 }, // 12
		cv::Point2f{ 847, 642 }, // 13
		cv::Point2f{ 866, 543 }, // 14
		cv::Point2f{ 886, 447 }, // 15
		cv::Point2f{ 895, 348 }, // 16
		cv::Point2f{ 174, 254 }, // 17
		cv::Point2f{ 236, 222 }, // 18
		cv::Point2f{ 303, 217 }, // 19
		cv::Point2f{ 373, 239 }, // 20
		cv::Point2f{ 431, 273 }, // 21
		cv::Point2f{ 571, 273 }, // 22
		cv::Point2f{ 631, 241 }, // 23
		cv::Point2f{ 701, 211 }, // 24
		cv::Point2f{ 766, 220 }, // 25
		cv::Point2f{ 828, 246 }, // 26
		cv::Point2f{ 500, 366 }, // 27
		cv::Point2f{ 500, 430 }, // 28
		cv::Point2f{ 500, 494 }, // 29
		cv::Point2f{ 500, 558 }, // 30
		cv::Point2f{ 439, 609 }, // 31
		cv::Point2f{ 471, 619 }, // 32
		cv::Point2f{ 500, 623 }, // 33
		cv::Point2f{ 533, 623 }, // 34
		cv::Point2f{ 563, 609 }, // 35
		cv::Point2f{ 210, 348 }, // 36
		cv::Point2f{ 272, 322 }, // 37
		cv::Point2f{ 342, 326 }, // 38
		cv::Point2f{ 406, 370 }, // 39
		cv::Point2f{ 334, 390 }, // 40
		cv::Point2f{ 266, 390 }, // 41
		cv::Point2f{ 596, 372 }, // 42
		cv::Point2f{ 660, 328 }, // 43
		cv::Point2f{ 730, 324 }, // 44
		cv::Point2f{ 792, 350 }, // 45
		cv::Point2f{ 736, 390 }, // 46
		cv::Point2f{ 668, 392 }, // 47
		cv::Point2f{ 354, 734 }, // 48
		cv::Point2f{ 398, 706 }, // 49
		cv::Point2f{ 446, 688 }, // 50
		cv::Point2f{ 500, 686 }, // 51
		cv::Point2f{ 556, 688 }, // 52
		cv::Point2f{ 602, 706 }, // 53
		cv::Point2f{ 648, 734 }, // 54
		cv::Point2f{ 602, 782 }, // 55
		cv::Point2f{ 556, 806 }, // 56
		cv::Point2f{ 500, 810 }, // 57
		cv::Point2f{ 446, 806 }, // 58
		cv::Point2f{ 398, 781 }, // 59
		cv::Point2f{ 350, 734 }, // 60
		cv::Point2f{ 446, 734 }, // 61
		cv::Point2f{ 500, 734 }, // 62
		cv::Point2f{ 556, 734 }, // 63
		cv::Point2f{ 650, 734 }, // 64
		cv::Point2f{ 0, 0 },     // 65
		cv::Point2f{ 100, 321 }, // 66
		cv::Point2f{ 180, 173 }, // 67
		cv::Point2f{ 260, 99 },  // 68
		cv::Point2f{ 340, 62 },  // 69
		cv::Point2f{ 420, 44 },  // 70
		cv::Point2f{ 500, 35 },  // 71
		cv::Point2f{ 580, 44 },  // 72
		cv::Point2f{ 660, 62 },  // 73
		cv::Point2f{ 740, 99 },  // 74
		cv::Point2f{ 820, 173 }, // 75
		cv::Point2f{ 900, 321 }  // 76
	};

	static const std::array<cv::Point2f, TARGET_DETAIL_LIMIT> TexMapTable =
	{
		cv::Point2f{ 0.33125f, 0.299219f },
		cv::Point2f{ 0.336719f, 0.327344f },
		cv::Point2f{ 0.34375f, 0.354688f },
		cv::Point2f{ 0.353125f, 0.382813f },
		cv::Point2f{ 0.360156f, 0.409375f },
		cv::Point2f{ 0.430469f, 0.452344f },
		cv::Point2f{ 0.45f, 0.4625f },
		cv::Point2f{ 0.474219f, 0.469531f },
		cv::Point2f{ 0.501563f, 0.475f },
		cv::Point2f{ 0.528906f, 0.469531f },
		cv::Point2f{ 0.546875f, 0.4625f },
		cv::Point2f{ 0.567188f, 0.452344f },
		cv::Point2f{ 0.63125f, 0.409375f },
		cv::Point2f{ 0.65f, 0.382813f },
		cv::Point2f{ 0.659375f, 0.354688f },
		cv::Point2f{ 0.666406f, 0.327344f },
		cv::Point2f{ 0.671094f, 0.299219f },
		cv::Point2f{ 0.410156f, 0.273438f },
		cv::Point2f{ 0.427344f, 0.264063f },
		cv::Point2f{ 0.446875f, 0.264063f },
		cv::Point2f{ 0.465625f, 0.269531f },
		cv::Point2f{ 0.482813f, 0.279688f },
		cv::Point2f{ 0.519531f, 0.279688f },
		cv::Point2f{ 0.536719f, 0.269531f },
		cv::Point2f{ 0.554688f, 0.264063f },
		cv::Point2f{ 0.574219f, 0.264063f },
		cv::Point2f{ 0.590625f, 0.273438f },
		cv::Point2f{ 0.501563f, 0.305469f },
		cv::Point2f{ 0.501563f, 0.322656f },
		cv::Point2f{ 0.501563f, 0.341406f },
		cv::Point2f{ 0.501563f, 0.358594f },
		cv::Point2f{ 0.484375f, 0.372656f },
		cv::Point2f{ 0.492188f, 0.375781f },
		cv::Point2f{ 0.501563f, 0.377344f },
		cv::Point2f{ 0.509375f, 0.375781f },
		cv::Point2f{ 0.517188f, 0.372656f },
		cv::Point2f{ 0.421094f, 0.300781f },
		cv::Point2f{ 0.438281f, 0.292969f },
		cv::Point2f{ 0.457031f, 0.294531f },
		cv::Point2f{ 0.475781f, 0.307031f },
		cv::Point2f{ 0.455469f, 0.311719f },
		cv::Point2f{ 0.435938f, 0.311719f },
		cv::Point2f{ 0.527344f, 0.307031f },
		cv::Point2f{ 0.54375f, 0.294531f },
		cv::Point2f{ 0.564063f, 0.292969f },
		cv::Point2f{ 0.580469f, 0.300781f },
		cv::Point2f{ 0.565625f, 0.311719f },
		cv::Point2f{ 0.546875f, 0.311719f },
		cv::Point2f{ 0.460156f, 0.407813f },
		cv::Point2f{ 0.472656f, 0.400781f },
		cv::Point2f{ 0.485938f, 0.396094f },
		cv::Point2f{ 0.501563f, 0.396094f },
		cv::Point2f{ 0.515625f, 0.396094f },
		cv::Point2f{ 0.528906f, 0.400781f },
		cv::Point2f{ 0.540625f, 0.407813f },
		cv::Point2f{ 0.528906f, 0.421875f },
		cv::Point2f{ 0.515625f, 0.428125f },
		cv::Point2f{ 0.501563f, 0.429688f },
		cv::Point2f{ 0.485938f, 0.428125f },
		cv::Point2f{ 0.472656f, 0.421875f },
		cv::Point2f{ 0.485938f, 0.40625f },
		cv::Point2f{ 0.501563f, 0.40625f },
		cv::Point2f{ 0.517188f, 0.40625f },
		cv::Point2f{ 0.517188f, 0.4125f },
		cv::Point2f{ 0.501563f, 0.4125f },
		cv::Point2f{ 0.485938f, 0.4125f },   // 65
		cv::Point2f{ 0.410156f, 0.255469f }, // 66
		cv::Point2f{ 0.427344f, 0.244531f }, // 67
		cv::Point2f{ 0.465625f, 0.241406f }, // 68
		cv::Point2f{ 0.501563f, 0.239844f }, // 69
		cv::Point2f{ 0.536719f, 0.241406f }, // 70
		cv::Point2f{ 0.574219f, 0.244531f }, // 71
		cv::Point2f{ 0.590625f, 0.255469f }, // 72
		cv::Point2f{ 0.536719f, 0.241406f }, // 73
		cv::Point2f{ 0.574219f, 0.244531f }, // 74
		cv::Point2f{ 0.590625f, 0.255469f }, // 75
		cv::Point2f{ 0.590625f, 0.255469f }  // 76
	};

	class Face {
	public:
		Face();
		~Face();
	public:
		void init(std::array<cv::Point2f, TARGET_DETAIL_LIMIT> pts);
		std::vector<std::array<cv::Point2f, TOTAL_VERTEX_POINTS>> getCoords();
		void updateCoords(ImagePoints points);
		std::array<cv::Point2f, TARGET_DETAIL_LIMIT> coords;
		std::vector<std::vector<std::vector<int>>> vertexIndex =
		{
			{
				/*CHIN*/      { 8, 57, 7, 58, 6, 59, 5, 48, 4 },
				/*CHEEK*/     { 41, 36, 1, 2, 3, 4, 48, 49, 31, 40 },
				/*BROW*/      { 1, 0, 36, 17, 37, 18, 38, 20, 39, 71 },
				/*NOSE*/      { 39, 71, 27, 28, 29, 30, 31, 40 },
				/*PHILTRUM*/  { 31, 30, 33, 51, 50, 49 },
				/*FOREHEAD*/  { 0, 66, 17, 67, 18, 68, 20, 70, 21, 71 },
				/*MOUTH*/     { 48, 49, 59, 50, 58, 51, 57 },
				/*EYE*/       { 36, 37, 41, 38, 40, 39 }
			},
			{
				/*CHIN*/      { 8, 57, 9, 56, 10, 55, 11, 54, 12 },
				/*CHEEK*/     { 46, 45, 15, 14, 13, 12, 54, 53, 35, 47 },
				/*BROW*/      { 15, 16, 45, 26, 44, 25, 43, 23, 42, 71 },
				/*NOSE*/      { 42, 71, 27, 28, 29, 30, 35, 47 },
				/*PHILTRUM*/  { 35, 30, 33, 51, 52, 53 },
				/*FOREHEAD*/  { 16, 76, 26, 75, 25, 74, 23, 72, 22, 71 },
				/*MOUTH*/     { 54, 53, 55, 52, 56, 51, 57 },
				/*EYE*/       { 42, 43, 47, 44, 46, 45 }
			}
		};
		cv::Point2f top;
		cv::Point2f bottom;
	};

	static const Face getFaceModel()
	{
		Face fm;
		fm.init(InitTable);
		return fm;
	};
}

