#pragma once

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <array>
#include <string>
#include <GLSL/GLSL.h>

using namespace Eigen;

class PointCalcUtil
{
public:
	PointCalcUtil();
	~PointCalcUtil();
	
	static double distanceBetween(double x1, double y1, double x2, double y2)
	{
		double x = x1 - x2;
		double y = y1 - y2;
		double dist;
		dist = pow(x, 2) + pow(y, 2);
		dist = sqrt(dist);

		return dist;
	}

	static double distanceBetween(const cv::Point2f& a, const cv::Point2f& b)
	{
		double x = a.x - b.x;
		double y = a.y - b.y;
		double dist;
		dist = pow(x, 2) + pow(y, 2);
		dist = sqrt(dist);

		return dist;
	}

	static double distanceBetween(const cv::Point3f& a, const cv::Point3f& b)
	{
		double x = a.x - b.x;
		double y = a.y - b.y;
		double z = a.z - b.z;
		double dist;
		dist = pow(x, 2) + pow(y, 2) + pow(z, 2);
		dist = sqrt(dist);

		return dist;
	}

	static double distanceBetween(const cv::Point2i& a, const cv::Point2i& b)
	{
		double x = a.x - b.x;
		double y = a.y - b.y;
		double dist;
		dist = pow(x, 2) + pow(y, 2);
		dist = sqrt(dist);

		return dist;
	}

	static cv::Point centerOf2Points(const cv::Point& a, const cv::Point& b) {
		cv::Point ret;
		ret.x = (a.x + b.x) / 2;
		ret.y = (a.y + b.y) / 2;
		return ret;
	}

	static cv::Point2f interPoint(const cv::Point2f& a, const cv::Point2f& b, float r) {
		cv::Point2f pt;
		pt.x = a.x*r + b.x*(1.f - r);
		pt.y = a.y*r + b.y*(1.f - r);
		return pt;
	}
	static float sq_norm(const cv::Point3f& a) {
		return a.x*a.x + a.y*a.y + a.z*a.z;
	}
	static float norm(const cv::Point3f& a) {
		return sqrt(sq_norm(a));
	}
	static float sq_norm(const cv::Point2f& a) {
		return a.x*a.x + a.y*a.y;
	}
	static float norm(const cv::Point2f& a) {
		return sqrt(sq_norm(a));
	}

	static float normalize(cv::Point3f& a) {
		float norm = PointCalcUtil::norm(a);
		if (norm > 10e-6)
			norm = (1.0f / norm);
		else
			norm = 0.0;
		a.x = (float)(a.x*norm);
		a.y = (float)(a.y*norm);
		a.z = (float)(a.z*norm);
		return norm;
	}

	static cv::Point3f cross(const cv::Point3f& a, const cv::Point3f& b) {
		return cv::Point3f(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
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

	static Eigen::Vector2f crossPointTwoLines(const Eigen::Vector2f& p1, const Eigen::Vector2f& p2, const Eigen::Vector2f& p3, const Eigen::Vector2f& p4)
	{
		float d = (p4[1] - p3[1])*(p2[0] - p1[0]) - (p4[0] - p3[0])*(p2[1] - p1[1]);
		if (d == 0)
			return Eigen::Vector2f();

		float ua;
		ua = ((p4[0] - p3[0])*(p1[1] - p3[1]) - (p4[1] - p3[1])*(p1[0] - p3[0])) / (float)d;
		Eigen::Vector2f pt;
		pt[0] = p1[0] + ua*(p2[0] - p1[0]);
		pt[1] = p1[1] + ua*(p2[1] - p1[1]);
		return pt;
	}

	static Vector2f getBisector(Vector2f v1, Vector2f v2)
	{
		return (v1.normalized() + v2.normalized()).normalized();
	}

	static Vector2f getVectorNormal(Vector2f v)
	{
		return Vector2f(v[1], -v[0]).normalized();
	}

	static char* StructUniformField1(std::string name, int index, std::string field)
	{
		return const_cast<char*>((boost::format("%s[%d].%s") % name % index % field).str().c_str());
	}

	static char* StructUniformField2(std::string name, int index1, std::string field1, int index2, std::string field2)
	{
		return const_cast<char*>((boost::format("%s[%d].%s[%d].%s") % name % index1 % field1 % index2 % field2).str().c_str());
	}

	template<std::size_t N>
	static void LoadLinesToUniform(std::array<Eigen::Vector2f, N>& linesArray, std::string uniformName, std::shared_ptr<cwc::glShader>& shader)
	{
		for (int i = 0; i < linesArray.size() - 1; ++i)
		{
			float length = (linesArray[i + 1] - linesArray[i]).norm();
			shader->setUniform2f(StructUniformField1(uniformName, i, "A"), linesArray[i][0], linesArray[i][1]);
			shader->setUniform2f(StructUniformField1(uniformName, i, "B"), linesArray[i + 1][0], linesArray[i + 1][1]);
			shader->setUniform1f(StructUniformField1(uniformName, i, "length"), length);
		}
	}

	static std::vector<cv::Point> makeNaturalContour(const std::vector<cv::Point>& points, double strength, std::vector<double>& offset_values)
	{
		const int inter_count = offset_values.size();
		std::vector<cv::Point> naturalPoints;
		naturalPoints.push_back(points[0]);
		for (int i = 1; i < points.size(); i++)
		{
			cv::Point pt1 = points[i - 1];
			cv::Point pt2 = points[i];
			double dist = PointCalcUtil::distanceBetween(pt1.x, pt1.y, pt2.x, pt2.y);
			double step = dist / (inter_count + 2);
			double dx = (double)(pt2.x - pt1.x) / dist;
			double dy = (double)(pt2.y - pt1.y) / dist;
			double nx = -dy;
			double ny = dx;
			double var_offset = strength * dist;
			for (int k = 0; k < inter_count; k++)
			{
				double dist_elem = step * (k + 1);
				double dx1 = dist_elem * dx;
				double dy1 = dist_elem * dy;
				cv::Point newPt(pt1.x + dx1 + nx * offset_values[k] * var_offset, pt1.y + dy1 + ny * offset_values[k] * var_offset);
				naturalPoints.push_back(newPt);
			}
			naturalPoints.push_back(pt2);
		}
		return naturalPoints;
	}

	static std::vector<cv::Point> makeSmallNaturalContour(const std::vector<cv::Point>& points, double strength)
	{
		const double arr[] = { 0.8, 0.2, 0.4, -0.6, -0.3, 0.2, -0.5, -0.2 };
		std::vector<double> vec(arr, arr + sizeof(arr) / sizeof(arr[0]));
		return makeNaturalContour(points, strength, vec);
	}
};
