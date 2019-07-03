#pragma once

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

class MaterialData
{
public:
	std::string diffuse_texture;
	std::string specular_texture;
	std::string specular_highlight_texture;
	std::string bump_texture;
	std::string emissive_texture;
	std::string normal_texture;
};

class GraphicsData
{
public:

	int numTriangles;

	size_t material_id;
	std::string material;

	MaterialData materialData;

	GLuint va = 0;
	GLuint la = 0;
	GLuint vb = 0;
	GLuint nb = 0;
	GLuint cb = 0;
	GLuint tb = 0;

	GLuint tangentb = 0;
	GLuint bitangentb = 0;

	std::string name;

	float minimumZ = 0;

};

class Mesh3D
{
public:
	int rows;
	int cols;
	std::vector<cv::Point2f> vertices;
	std::vector<cv::Point2f> uvs;
	std::vector<ushort> idxs;

	Mesh3D() {}
	~Mesh3D() {}

	Mesh3D(const Mesh3D& t)
	{
		rows = t.rows;
		cols = t.cols;
		vertices.resize(t.vertices.size());
		uvs.resize(t.uvs.size());
		idxs.resize(t.idxs.size());
		std::copy(t.vertices.begin(), t.vertices.end(), vertices.begin());
		std::copy(t.uvs.begin(), t.uvs.end(), uvs.begin());
		std::copy(t.idxs.begin(), t.idxs.end(), idxs.begin());
	}

	void release()
	{
		vertices.clear();
		uvs.clear();
		idxs.clear();
	}
};

class Mesh2D
{
public:
	Mesh2D();
	Mesh2D(const Mesh2D& t);
	~Mesh2D();

	Mesh2D& operator=(const Mesh2D& t);

	void init(int rows, int cols);
	void setPoint(int ix,int iy, float x, float y);
	cv::Point2f getPoint(int ix, int iy) const;
	cv::Size size() const;
	void reset();
private:
	cv::Mat1f m_xmat, m_ymat;
	std::vector<cv::Point2f> m_vecpoints;
};

class Mesh3DMaker
{
public:
	Mesh3DMaker();
	~Mesh3DMaker();
	
	void init(const cv::Mat& image, const cv::Rect& viewRegion, const Mesh2D& mesh);
	void setDstMesh(const Mesh2D& dst);

	cv::Rect getViewRegion() const { return m_viewRegion; }
	Mesh3D getMesh();

protected:
	int m_rows;
	int m_cols;

	cv::Mat m_image;
	cv::Size m_imgsize;
	cv::Rect m_viewRegion;

	Mesh2D m_srcMesh;
	Mesh2D m_dstMesh;
};
