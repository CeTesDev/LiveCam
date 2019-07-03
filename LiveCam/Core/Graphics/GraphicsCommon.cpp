#include "GraphicsCommon.h"

Mesh2D::Mesh2D(const Mesh2D& t) {
	m_xmat = t.m_xmat;
	m_ymat = t.m_ymat;
}
Mesh2D& Mesh2D::operator=(const Mesh2D& t) {
	m_xmat = t.m_xmat;
	m_ymat = t.m_ymat;
	return *this;
}

Mesh2D::Mesh2D()
{

}
Mesh2D::~Mesh2D()
{

}
void Mesh2D::init(int rows, int cols)
{
	m_xmat.create(rows, cols);
	m_ymat.create(rows, cols);

	m_xmat = 0;
	m_ymat = 0;
	m_vecpoints.clear();
}
void Mesh2D::reset() {
	m_xmat = 0;
	m_ymat = 0;
}

cv::Size Mesh2D::size() const {
	return m_xmat.size();
}

void Mesh2D::setPoint(int ix, int iy, float x, float y)
{
	m_xmat[iy][ix] = x;
	m_ymat[iy][ix] = y;
}
cv::Point2f Mesh2D::getPoint(int ix, int iy) const
{
	return cv::Point2f(m_xmat[iy][ix], m_ymat[iy][ix]);
}

Mesh3DMaker::Mesh3DMaker()
{
}

Mesh3DMaker::~Mesh3DMaker()
{
}

void Mesh3DMaker::init(const cv::Mat& image, const cv::Rect& viewRegion, const Mesh2D& mesh)
{
	m_srcMesh = mesh;

	m_image = image;
	m_imgsize = image.size();
	m_viewRegion = viewRegion;
	m_rows = m_srcMesh.size().height - 1;
	m_cols = m_srcMesh.size().width - 1;

	m_srcMesh = mesh;
	m_dstMesh = m_srcMesh;
}

void Mesh3DMaker::setDstMesh(const Mesh2D& dst)
{
	m_dstMesh = dst;
}

Mesh3D Mesh3DMaker::getMesh()
{
	Mesh3D mesh;

	int count = (m_rows + 1)*(m_cols + 1);

	mesh.rows = m_rows;
	mesh.cols = m_cols;
	mesh.vertices.resize(count);
	mesh.uvs.resize(count);
	mesh.idxs.resize(m_rows*m_cols * 6);

	float aspectratio = 1.0f;
	float scalex = 1.0f / (float)m_imgsize.width;
	float scaley = 1.0f / (float)m_imgsize.height;

	for (int i = 0; i < count; i++)
	{
		int ix = i % (m_cols + 1);
		int iy = i / (m_cols + 1);
		mesh.uvs[i] = m_srcMesh.getPoint(ix, iy);
		mesh.uvs[i].x *= scalex;
		mesh.uvs[i].y = mesh.uvs[i].y * scaley;

		mesh.vertices[i] = m_dstMesh.getPoint(ix, iy);
		mesh.vertices[i].x *= scalex;
		mesh.vertices[i].y = (m_imgsize.height - mesh.vertices[i].y - 1) * scaley*aspectratio;

		mesh.vertices[i].y = 1.f - mesh.vertices[i].y;
	}

	int iquad = 0;

	for (int iy = 0; iy < m_rows; iy++)
	{
		for (int ix = 0; ix < m_cols; ix++, iquad += 6)
		{
			int i00 = iy*(m_cols + 1) + ix;
			int i01 = iy*(m_cols + 1) + ix + 1;
			int i10 = (iy + 1)*(m_cols + 1) + ix;
			int i11 = (iy + 1)*(m_cols + 1) + ix + 1;
			mesh.idxs[iquad] = i00;
			mesh.idxs[iquad + 1] = i01;
			mesh.idxs[iquad + 2] = i11;
			mesh.idxs[iquad + 3] = i00;
			mesh.idxs[iquad + 4] = i11;
			mesh.idxs[iquad + 5] = i10;
		}
	}
	
	return mesh;
}
