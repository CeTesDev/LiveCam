#pragma once

#include "GraphicsLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

void ProcessOneShape(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape, std::vector<Vector3f>& vb,
	std::vector<Vector3f>& cb,
	std::vector<Vector3f>& nb,
	std::vector<Vector2f>& tb,
	std::vector<Vector3f>& tangentb,
	std::vector<Vector3f>& bitangentb,
	std::vector<int>& indexes,
	std::array<float, 3>& bmin, std::array<float, 3>& bmax, float& objectZmin, float extraScale, Vector3f extraShift, Matrix3f extraRotateMatrix)
{
	indexes.reserve(shape.mesh.indices.size());
	vb.reserve(shape.mesh.indices.size());
	cb.reserve(shape.mesh.indices.size());
	nb.reserve(shape.mesh.indices.size());
	tb.reserve(shape.mesh.indices.size());
	tangentb.reserve(shape.mesh.indices.size());
	bitangentb.reserve(shape.mesh.indices.size());

	for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
		tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
		tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
		tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

		indexes.push_back(idx0.vertex_index);
		indexes.push_back(idx1.vertex_index);
		indexes.push_back(idx2.vertex_index);

		Vector2f tc[3];
		if (attrib.texcoords.size() > 0) {
			assert(attrib.texcoords.size() > 2 * idx0.texcoord_index + 1);
			assert(attrib.texcoords.size() > 2 * idx1.texcoord_index + 1);
			assert(attrib.texcoords.size() > 2 * idx2.texcoord_index + 1);
			int index1 = 2 * idx0.texcoord_index;
			int index2 = 2 * idx1.texcoord_index;
			int index3 = 2 * idx2.texcoord_index;
			tc[0][0] = attrib.texcoords[index1];
			tc[0][1] = 1.0f - attrib.texcoords[index1 + 1];
			tc[1][0] = attrib.texcoords[index2];
			tc[1][1] = 1.0f - attrib.texcoords[index2 + 1];
			tc[2][0] = attrib.texcoords[index3];
			tc[2][1] = 1.0f - attrib.texcoords[index3 + 1];

		}
		else {
			tc[0][0] = 0.0f;
			tc[0][1] = 0.0f;
			tc[1][0] = 0.0f;
			tc[1][1] = 0.0f;
			tc[2][0] = 0.0f;
			tc[2][1] = 0.0f;
		}

		Vector3f v[3];
		int f0 = idx0.vertex_index;
		int f1 = idx1.vertex_index;
		int f2 = idx2.vertex_index;
		assert(f0 >= 0);
		assert(f1 >= 0);
		assert(f2 >= 0);

		Eigen::Vector3f A = extraRotateMatrix * Eigen::Vector3f(
			attrib.vertices[3 * f0 + 0],
			attrib.vertices[3 * f0 + 1],
			attrib.vertices[3 * f0 + 2]) * extraScale + extraShift;

		Eigen::Vector3f B = extraRotateMatrix * Eigen::Vector3f(
			attrib.vertices[3 * f1 + 0],
			attrib.vertices[3 * f1 + 1],
			attrib.vertices[3 * f1 + 2]) * extraScale + extraShift;

		Eigen::Vector3f C = extraRotateMatrix * Eigen::Vector3f(
			attrib.vertices[3 * f2 + 0],
			attrib.vertices[3 * f2 + 1],
			attrib.vertices[3 * f2 + 2]) * extraScale + extraShift;

		for (int k = 0; k < 3; k++)
		{
			v[0][k] = A[k];
			v[1][k] = B[k];
			v[2][k] = C[k];

			bmin[k] = std::min(v[0][k], bmin[k]);
			bmin[k] = std::min(v[1][k], bmin[k]);
			bmin[k] = std::min(v[2][k], bmin[k]);
			bmax[k] = std::max(v[0][k], bmax[k]);
			bmax[k] = std::max(v[1][k], bmax[k]);
			bmax[k] = std::max(v[2][k], bmax[k]);
		}

		objectZmin = std::min(objectZmin, v[0][2]);
		objectZmin = std::min(objectZmin, v[1][2]);
		objectZmin = std::min(objectZmin, v[2][2]);

		for (int k = 0; k < 3; k++) {

			vb.push_back({ v[k][0], v[k][1], v[k][2] });

			tb.push_back({ tc[k][0], tc[k][1] });
		}
	}

	Eigen::Vector3f autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };

	for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
		tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
		tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
		tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

		int index = nb.size();
		Vector3f v[3] = { vb[index], vb[index + 1], vb[index + 2] };
		Vector2f tc[3] = { tb[index], tb[index + 1], tb[index + 2] };

		Vector3f n[3];
		if (attrib.normals.size() > 0) {
			int f0 = idx0.normal_index;
			int f1 = idx1.normal_index;
			int f2 = idx2.normal_index;
			assert(f0 >= 0);
			assert(f1 >= 0);
			assert(f2 >= 0);

			Eigen::Vector3f A = extraRotateMatrix * Eigen::Vector3f(
				attrib.normals[3 * f0 + 0],
				attrib.normals[3 * f0 + 1],
				attrib.normals[3 * f0 + 2]) * extraScale + extraShift;

			Eigen::Vector3f B = extraRotateMatrix * Eigen::Vector3f(
				attrib.normals[3 * f1 + 0],
				attrib.normals[3 * f1 + 1],
				attrib.normals[3 * f1 + 2]) * extraScale + extraShift;

			Eigen::Vector3f C = extraRotateMatrix * Eigen::Vector3f(
				attrib.normals[3 * f2 + 0],
				attrib.normals[3 * f2 + 1],
				attrib.normals[3 * f2 + 2]) * extraScale + extraShift;

			for (int k = 0; k < 3; k++) {
				n[0][k] = A[k];
				n[1][k] = B[k];
				n[2][k] = C[k];
			}
		}
		else
		{
			n[0] = (v[0] + autoShift).normalized();
			n[1] = (v[1] + autoShift).normalized();
			n[2] = (v[2] + autoShift).normalized();
		}

		Vector3f deltaPos1 = v[1] - v[0];
		Vector3f deltaPos2 = v[2] - v[0];

		Vector2f deltaUV1 = tc[1] - tc[0];
		Vector2f deltaUV2 = tc[2] - tc[0];

		Vector3f tangent = (deltaPos1 * deltaUV2[1] - deltaPos2 * deltaUV1[1]).normalized();
		Vector3f bitangent = (deltaPos2 * deltaUV1[0] - deltaPos1 * deltaUV2[0]).normalized();

		if (n[0].dot(tangent.cross(bitangent)) < 0)
		{
			tangent = -tangent;
		}

		for (int k = 0; k < 3; k++) {
			nb.push_back({ n[k][0], n[k][1], n[k][2] });

			tangentb.push_back(tangent);

			bitangentb.push_back(bitangent);
		}
	}
}

std::string GraphicsLoader::GetBaseDir(const std::string &filepath) {
	if (filepath.find_last_of("/\\") != std::string::npos)
		return filepath.substr(0, filepath.find_last_of("/\\"));
	return "";
}

bool GraphicsLoader::FileExists(const std::string &abs_filename) {
	bool ret;
	FILE *fp;
	fopen_s(&fp, abs_filename.c_str(), "rb");
	if (fp) {
		ret = true;
		fclose(fp);
	}
	else {
		ret = false;
	}

	return ret;
}

Vector4f GraphicsLoader::CalculateQuat(Vector3f vecFrom, Vector3f vecTo)
{
	Vector4f result;
	Vector3f crossProductV = vecFrom.cross(vecTo);
	float len = crossProductV.norm();
	if (len < 0.05)
	{
		len = 0.05;
	}
	crossProductV /= len;

	float dotProductF = vecFrom.dot(vecTo);

	float lenFrom = vecFrom.norm();
	if (lenFrom < 0.05)
	{
		lenFrom = 0.05;
	}

	float lenTo = vecTo.norm();
	if (lenTo < 0.05)
	{
		lenTo = 0.05;
	}

	float alpha = acos(dotProductF / (lenTo * lenFrom));

	float sinHalfAlpha = sin(alpha / 2);
	float cosHalfAlpha = cos(alpha / 2);

	result << crossProductV * sinHalfAlpha, cosHalfAlpha;

	return result;
}

Vector3f GraphicsLoader::RotateByQuat(Vector3f vec, Vector4f quat)
{
	Vector3f u;
	u[0] = quat[0];
	u[1] = quat[1];
	u[2] = quat[2];
	float s = quat[3];

	Vector3f result;

	float dotuv = u.dot(vec);

	float dotuu = u.dot(u);

	Vector3f crossProductuv = u.cross(vec);

	result = 2 * dotuv * u + (s*s - dotuu) * vec + 2 * s * crossProductuv;

	return result;
}

void GraphicsLoader::loadObjectsNames(std::vector<std::string> &objFiles, std::vector<std::string>& objectsNames)
{
	for (auto &file : objFiles)
	{

		ifstream ifs(file);

		if (!ifs.good())
		{
			return;
		}

		string name;
		name.reserve(32);

		string line;
		line.reserve(255);

		while (ifs.good())
		{
			getline(ifs, line);

			switch (line[0])
			{
			case 'g':
			case 'o':
			{
				if (line[1] == ' ')
				{
					auto offset = line.find_first_of(' ', 2);
					std::string name = line.substr(2, offset - 2);
					if (name != "default")
					{
						objectsNames.push_back(name);
					}
				}
				break;
			}
			}
		}

		ifs.close();

	}
}

void GraphicsLoader::loadMTLnames(std::vector<std::string> &objFiles, std::vector<GraphicsData>& drawObjects)
{
	for (auto &file : objFiles)
	{

		ifstream ifs(file);

		if (!ifs.good())
		{
			return;
		}

		string name;
		name.reserve(32);

		string line;
		line.reserve(255);

		string lastMeshName;

		while (ifs.good())
		{
			getline(ifs, line);

			switch (line[0])
			{
			case 'g':
			case 'o':
			{
				if (line[1] == ' ')
				{
					auto offset = line.find_first_of(' ', 2);
					lastMeshName = line.substr(2, offset - 2);
				}
				break;
			}

			case 'u':
			{
				if (line.substr(0, 7) == "usemtl " && !lastMeshName.empty())
				{
					auto iter = std::find_if(drawObjects.begin(), drawObjects.end(), [lastMeshName](GraphicsData& E) -> bool
					{
						return E.name == lastMeshName;
					});

					if (iter != drawObjects.end())
					{
						iter->material = line.substr(7);
					}
				}
				break;
			}
			}
		}

		ifs.close();

	}
}

void GraphicsLoader::LoadObj(std::vector<std::string> &objFiles, std::vector<GraphicsData>& drawObjects,
	std::array<float, 3>& bmin, std::array<float, 3>& bmax, float extraScale, Vector3f extraShift, Matrix3f extraRotateMatrix, std::vector<ObjectData>* outputBuffers)
{
	bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

	for (auto &file : objFiles)
	{
		float objectZmin = std::numeric_limits<float>::max();

		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;

		std::string base_dir = GetBaseDir(file);
		base_dir += "/";

		std::vector<tinyobj::material_t> materials;
		std::string err;
		std::string warn;
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file.data(), base_dir.c_str());
		if (!warn.empty()) {
			std::cout << warn << std::endl;
		}
		if (!err.empty()) {
			std::cerr << err << std::endl;
		}

		if (!ret) {
			std::cerr << "Failed to load " << file << std::endl;
			continue;
		}
		if (outputBuffers)
		{
			outputBuffers->resize(shapes.size());
		}

		for (size_t s = 0; s < shapes.size(); s++)
		{
			GraphicsData o;

			o.name = shapes[s].name;
			/*o.materialData.bump_texture = materials[s].bump_texname;
			o.materialData.diffuse_texture = materials[s].diffuse_texname;
			o.materialData.emissive_texture = materials[s].emissive_texname;
			o.materialData.normal_texture = materials[s].normal_texname;
			o.materialData.specular_highlight_texture = materials[s].specular_highlight_texname;
			o.materialData.specular_texture = materials[s].specular_texname;*/
			
			std::vector<Vector3f> vb;
			std::vector<Vector3f> cb;
			std::vector<Vector3f> nb;
			std::vector<Vector2f> tb;

			std::vector<Vector3f> tangentb;
			std::vector<Vector3f> bitangentb;

			std::vector<int> indexes;

			ProcessOneShape(attrib, shapes[s], vb, cb, nb, tb, tangentb, bitangentb, indexes, bmin, bmax, objectZmin, extraScale, extraShift, extraRotateMatrix);
			
			if (vb.size() > 0) {
				glGenBuffers(1, &o.vb);
				glBindBuffer(GL_ARRAY_BUFFER, o.vb);
				glBufferData(GL_ARRAY_BUFFER, vb.size() * sizeof(Vector3f), vb.data(),
					GL_STATIC_DRAW);
				o.numTriangles = vb.size() / 3;

				glGenVertexArrays(1, &o.va);
				glGenBuffers(1, &o.la);
				glBindVertexArray(o.va);
				glBindBuffer(GL_ARRAY_BUFFER, o.la);
				glBufferData(GL_ARRAY_BUFFER, vb.size() * sizeof(Vector3f), vb.data(), GL_STATIC_DRAW);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(Vector3f), (void*)0);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(Vector3f), (void*)(3 * sizeof(Vector3f)));
				glEnableVertexAttribArray(1);
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(Vector3f), (void*)(6 * sizeof(Vector3f)));
				glEnableVertexAttribArray(2);
			}

			if (cb.size() > 0) {
				glGenBuffers(1, &o.cb);
				glBindBuffer(GL_ARRAY_BUFFER, o.cb);
				glBufferData(GL_ARRAY_BUFFER, cb.size() * sizeof(Vector3f), cb.data(), GL_STATIC_DRAW);

			}

			if (nb.size() > 0) {
				glGenBuffers(1, &o.nb);
				glBindBuffer(GL_ARRAY_BUFFER, o.nb);
				glBufferData(GL_ARRAY_BUFFER, nb.size() * sizeof(Vector3f), nb.data(), GL_STATIC_DRAW);

			}

			if (tangentb.size() > 0) {
				glGenBuffers(1, &o.tangentb);
				glBindBuffer(GL_ARRAY_BUFFER, o.tangentb);
				glBufferData(GL_ARRAY_BUFFER, tangentb.size() * sizeof(Vector3f), tangentb.data(), GL_STATIC_DRAW);

			}

			if (bitangentb.size() > 0) {
				glGenBuffers(1, &o.bitangentb);
				glBindBuffer(GL_ARRAY_BUFFER, o.bitangentb);
				glBufferData(GL_ARRAY_BUFFER, bitangentb.size() * sizeof(Vector3f), bitangentb.data(), GL_STATIC_DRAW);

			}

			if (tb.size() > 0) {
				glGenBuffers(1, &o.tb);
				glBindBuffer(GL_ARRAY_BUFFER, o.tb);
				glBufferData(GL_ARRAY_BUFFER, tb.size() * sizeof(Vector2f), tb.data(), GL_STATIC_DRAW);

			}

			if (outputBuffers)
			{
				(*outputBuffers)[s].vb = vb;
				(*outputBuffers)[s].nb = nb;
				(*outputBuffers)[s].tangentb = tangentb;
				(*outputBuffers)[s].bitangentb = bitangentb;
				(*outputBuffers)[s].indexes = indexes;
			}

			drawObjects.push_back(o);
			o.minimumZ = objectZmin;
		}

	}
}

void GraphicsLoader::LoadObjData(const std::string &file, std::vector<ObjectData>& objDataArr, std::array<float, 3>& bmin, std::array<float, 3>& bmax, float extraScale, Vector3f extraShift, Matrix3f extraRotateMatrix)
{
	bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

	float objectZmin = std::numeric_limits<float>::max();

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;

	std::string base_dir = GetBaseDir(file);
	base_dir += "/";

	std::vector<tinyobj::material_t> materials;
	std::string err;
	std::string warn;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file.data(), base_dir.c_str());
	if (!warn.empty()) {
		std::cout << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << err << std::endl;
	}

	if (!ret) {
		std::cerr << "Failed to load " << file << std::endl;
		return;
	}

	for (size_t s = 0; s < shapes.size(); s++)
	{
		ObjectData o;

		ProcessOneShape(attrib, shapes[s], o.vb, o.cb, o.nb, o.tb, o.tangentb, o.bitangentb, o.indexes, bmin, bmax, objectZmin, extraScale, extraShift, extraRotateMatrix);

		objDataArr.push_back(o);

	}

}

GraphicsData GraphicsLoader::CreateQuadModel()
{
	GraphicsData obj;

	std::array<Vector3f, 6> vb =
	{
		Vector3f(-1, -1, 0), Vector3f(-1, 1, 0), Vector3f(1, 1, 0),
		Vector3f(1, 1, 0), Vector3f(1, -1, 0), Vector3f(-1, -1, 0)
	};
	std::array<Vector2f, 6> tb =
	{
		Vector2f(0, 0), Vector2f(0, 1), Vector2f(1, 1),
		Vector2f(1, 1), Vector2f(1, 0), Vector2f(0, 0)
	};

	glGenBuffers(1, &obj.vb);
	glBindBuffer(GL_ARRAY_BUFFER, obj.vb);
	glBufferData(GL_ARRAY_BUFFER, vb.size() * sizeof(Vector3f), vb.data(), GL_STATIC_DRAW);
	obj.numTriangles = 2;

	glGenBuffers(1, &obj.tb);
	glBindBuffer(GL_ARRAY_BUFFER, obj.tb);
	glBufferData(GL_ARRAY_BUFFER, tb.size() * sizeof(Vector2f), tb.data(), GL_STATIC_DRAW);

	return obj;
}

Vector3f loadVector3f(char *dataBegin)
{
	Vector3f result;
	char *begin = dataBegin;
	char *end = dataBegin;

	for (int i = 0; i < 3; ++i)
	{
		while (*end != ' ' && *end != 0) ++end;
		try
		{
			result[i] = stof(string(begin, end));
		}
		catch (const invalid_argument&)
		{
			throw std::exception(("Invalid OBJ file line: " + string(dataBegin)).c_str());
		}

		begin = ++end;
	}
	return result;
}

boost::property_tree::ptree GraphicsLoader::convertMTLtoPTree(const std::string &path, const std::string &resourcesRoot)
{
	boost::property_tree::ptree MTLs;

	ifstream ifs(path);

	if (!ifs.good())
	{
		return MTLs;
	}

	fs::path folder = resourcesRoot;

	boost::property_tree::ptree newMTL;

	string name;
	name.reserve(32);

	string line;
	line.reserve(255);

	bool isEmptyMTL = true;

	while (ifs.good())
	{
		getline(ifs, line);

		switch (line[0])
		{
		case 'n':
		{
			if (line.substr(0, 6) == "newmtl")
			{
				if (!isEmptyMTL)
				{
					MTLs.put_child(name, newMTL);
				}
				name = line.substr(7);
				newMTL.clear();
				isEmptyMTL = true;
			}
			break;
		}

		case 'K':
		{
			boost::property_tree::ptree vector3fTree;

			Vector3f vector3f;

			try
			{
				vector3f = loadVector3f(const_cast<char*>(line.c_str()) + 3);
			}
			catch (std::exception&)
			{
				break;
			}

			for (int i = 0; i < 3; ++i)
			{
				boost::property_tree::ptree vector3fTreeChild;
				vector3fTreeChild.put("", vector3f[i]);
				vector3fTree.push_back(make_pair("", vector3fTreeChild));
			}

			switch (line[1])
			{
			case 'a':
			{
				newMTL.put_child("ambientLight", vector3fTree);
				isEmptyMTL = false;
				break;
			}
			case 'd':
			{
				newMTL.put_child("diffuseLight", vector3fTree);
				isEmptyMTL = false;
				break;
			}
			case 's':
			{
				newMTL.put_child("specularLight", vector3fTree);
				isEmptyMTL = false;
				break;
			}
			}
			break;
		}

		case 'N':
		{
			if (line[1] == 's')
			{
				newMTL.put("specularPower", line.substr(3));
				isEmptyMTL = false;
			}
			break;
		}

		case 'm':
		{
			if (line.substr(0, 5) == "map_K")
			{
				switch (line[5])
				{
				case 'a':
				case 'd':
				{
					newMTL.put("Texture", (folder / line.substr(line[7] == '/' || line[7] == '\\' ? 8 : 7)).string());
					isEmptyMTL = false;
					break;
				}
				
				}
			}
			break;
		}

		case 'b':
		{
			if (line.substr(0, 4) == "bump")
			{
				newMTL.put("normalMap", (folder / line.substr(line[5] == '/' || line[5] == '\\' ? 6 : 5)).string());
				isEmptyMTL = false;
			}
			break;
		}

		}
	}

	ifs.close();

	if (!isEmptyMTL)
	{
		MTLs.put_child(name, newMTL);
	}

	return MTLs;
}