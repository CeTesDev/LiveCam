#include <Widgets/FxWidgetFaceMask.h>
#include <Graphics/GraphicsLoader.h>

#include <QDebug>

const std::string FxWidgetFaceMask::TYPE_NAME = "FxWidgetFaceMask";

std::string FxWidgetFaceMask::getTypeName()
{
	return TYPE_NAME;
}

void FxWidgetFaceMask::onInputFrameResized()
{
	GraphicsModel::onInputFrameResized();
}

FxWidgetFaceMask::FxWidgetFaceMask()
{
	bmin = { 0, 0, 0 };
	bmax = { 0, 0, 0 };
}

void FxWidgetFaceMask::loadFromJSON(boost::property_tree::ptree& modelRecord)
{
	GraphicsModel::loadFromJSON(modelRecord);

	auto extraShiftVector = modelRecord.get_child_optional("modelShift");
	modelShift = extraShiftVector ? JSONVectorReader::readVector3f(extraShiftVector.get()) : Vector3f(0, 0, 0);

	modelScale = modelRecord.get<float>("modelScale", 1);

	auto rotateMatrixRecord = modelRecord.get_child_optional("modelRotation");
	extraRotateMatrix = rotateMatrixRecord ? JSONVectorReader::readMatrix3f(rotateMatrixRecord.get()) : Matrix3f::Identity();

	modelPath = modelRecord.get<std::string>("OBJ", "");

	bonesPath = modelRecord.get<std::string>("bones", "");

	weightsPath = modelRecord.get<std::string>("weights", "");

	bonesStretching = modelRecord.get<bool>("bonesStretching", true);
}

bool FxWidgetFaceMask::reloadOBJ()
{
	objects.clear();

	float extraScale;
	Vector3f extraShift;

	extraScale = 1;
	extraShift = Vector3f(0, 0, 0);
	

	std::vector<std::string> OBJs;

	
	OBJs.push_back((resourceManager.resourceRoot / modelPath).string());

	GraphicsLoader::LoadObj(OBJs, objects, bmin, bmax, extraScale, extraShift, extraRotateMatrix, &initialMeshes);

	transformedMeshes.resize(initialMeshes.size());
	for (int i = 0; i < initialMeshes.size(); ++i)
	{
		transformedMeshes[i].vb.resize(initialMeshes[i].vb.size());
		transformedMeshes[i].nb.resize(initialMeshes[i].nb.size());
		transformedMeshes[i].tangentb.resize(initialMeshes[i].tangentb.size());
		transformedMeshes[i].bitangentb.resize(initialMeshes[i].bitangentb.size());
	}

	GraphicsLoader::loadMTLnames(OBJs, objects);
	
	return true;
}

void FxWidgetFaceMask::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	auto IDs = suit.get_child_optional("renderParamsIDs");
	if (IDs)
	{
		objectRenderParams[targetIndex].clear();
		for (auto &ID : IDs.get())
		{
			objectRenderParams[targetIndex].push_back(commonRenderParams[ID.second.get_value<size_t>()]);
		}
	}

	auto modelTextures = suit.get_child_optional("modelTextures");
	if (modelTextures)
	{
		texturesPaths[targetIndex].clear();

		for (auto &texture : modelTextures.get())
		{
			texturesPaths[targetIndex].push_back(texture.second.get_value<std::string>());
		}

		texturesIDs[targetIndex].clear();

		if (loadTexturesImmediately)
		{
			for (auto &texture : texturesPaths[targetIndex])
			{
				texturesIDs[targetIndex].push_back(resourceManager.loadTexture(texture).ID);
			}
		
			for (auto &params : objectRenderParams[targetIndex])
			{
				if (!params->normalMap.empty())
				{
					params->normalMapID = resourceManager.loadTexture(params->normalMap).ID;
				}
			}
		}
	}
}

bool FxWidgetFaceMask::load()
{
	GraphicsModel::load();

	if  (!reloadOBJ())
	{
		return false;
	}

	

	for (int target = 0; target < MAX_TO_TRACK; ++target)
	{
		texturesIDs[target].clear();
		
		for (auto &texture : texturesPaths[target])
		{
			texturesIDs[target].push_back(resourceManager.loadTexture(texture).ID);
		}

		for (auto &params : objectRenderParams[target])
		{
			if (!params->normalMap.empty())
			{
				params->normalMapID = resourceManager.loadTexture(params->normalMap).ID;
			}
		}
	}

	pt::ptree bonesJson, weightJson;
	pt::json_parser::read_json(bonesPath, bonesJson);
	pt::json_parser::read_json(weightsPath, weightJson);

	skeleton.load(bonesJson, weightJson, initialMeshes[0]);

	return true;
}

void FxWidgetFaceMask::transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	float width = externalRenderParams.trackedWidth;
	float height = externalRenderParams.trackedHeight;
	float zFar = externalRenderParams.zFar;
	float zNear = externalRenderParams.zNear;

	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = 2 / width;
	projection(1, 1) = -2 / height;
	projection(3, 3) = 1;
	projection(0, 3) = -1;
	projection(1, 3) = 1;

	
	projection(2, 2) = 2 / (zNear - zFar);
	projection(2, 3) = (zFar + zNear) / (zNear - zFar);

	renderParams.projection = projection;
	Vector2f noseTop = { target.pts[27 * 2], target.pts[27 * 2 + 1] };
	Vector2f chin = { target.pts[30 * 2], target.pts[30 * 2 + 1] };

	boneTransformations[target.pointId] = skeleton.calcTransformations(target, bonesStretching);

	const float STD_BODY_DISTANCE = 2.2f;

	Vector3f modelSize = { bmax[0] - bmin[0],  bmax[1] - bmin[1], bmax[2] - bmin[2] };
	float zScale = 1 / modelSize[2];

#ifndef SKELETAL_ANIMATION_TEST
	float skeletonScale = (noseTop - chin).norm() / skeleton.chinToNoseTopLength;
	float scale = skeletonScale * modelScale;

	Vector3f modelRelativeShift = modelShift.cwiseProduct(modelSize);
	Matrix4f modelView = Affine3f(Translation3f(modelRelativeShift)).matrix();

	Matrix4f compressingModelView = Affine3f(Scaling(scale, -scale, zScale)).matrix() * modelView;
	modelView = Affine3f(Scaling(scale, -scale, scale)).matrix() * modelView;

	Vector3f bodyPointsShift; bodyPointsShift << noseTop, -STD_BODY_DISTANCE;
	Matrix4f shiftMatrix = Affine3f(Translation3f(bodyPointsShift)).matrix();

	modelView = shiftMatrix * modelView;
	compressingModelView = shiftMatrix * compressingModelView;

#else
	float scale = 700 / modelSize[1];
	Vector3f autoShift = { -(bmax[0] + bmin[0]) / 2, -(bmax[1] + bmin[1]) / 2, -(bmax[2] + bmin[2]) / 2 };
	Matrix4f modelView = Affine3f(Translation3f(autoShift)).matrix();

	Matrix4f compressingModelView = Affine3f(Scaling(scale, -scale, zScale)).matrix() * modelView;
	modelView = Affine3f(Scaling(scale, -scale, scale)).matrix() * modelView;

	Vector3f shift = { width / 2, height / 2, -STD_BODY_DISTANCE };
	Matrix4f shiftMatrix = Affine3f(Translation3f(shift)).matrix();

	modelView = shiftMatrix * modelView;
	compressingModelView = shiftMatrix * compressingModelView;
#endif

	renderParams.modelView = modelView;
	renderParams.additionalMatrices4[0] = compressingModelView;
	renderParams.rotationMatrix = Matrix3f::Identity();
}

void FxWidgetFaceMask::drawBones(cv::Mat frame, float scaleBones, Eigen::Vector2f moveBones)
{
	static const std::array<cv::Scalar, BONES_COUNT> pairColors = {
	};

	int i = 0;
	for (auto& bone : skeleton.bones)
	{
		Eigen::Vector2f A = bone.begin * scaleBones + moveBones;
		Eigen::Vector2f B = bone.end * scaleBones + moveBones;

		A[1] = frame.rows - A[1];
		B[1] = frame.rows - B[1];

		Eigen::Vector2f V = (A - B) / 5;

		Eigen::Vector2f L, R;

		L << cos(M_PI_4) * V[0] - sin(M_PI_4) * V[1],
			 sin(M_PI_4) * V[0] + cos(M_PI_4) * V[1];

		R << cos(-M_PI_4) * V[0] - sin(-M_PI_4) * V[1],
			sin(-M_PI_4) * V[0] + cos(-M_PI_4) * V[1];

		L += B;
		R += B;

		cv::line(frame, { (int)A[0], (int)A[1] }, { (int)B[0], (int)B[1] }, pairColors[i], 4);
		cv::line(frame, { (int)L[0], (int)L[1] }, { (int)B[0], (int)B[1] }, pairColors[i], 4);
		cv::line(frame, { (int)R[0], (int)R[1] }, { (int)B[0], (int)B[1] }, pairColors[i], 4);

		++i;
	}
}

void FxWidgetFaceMask::draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	for (size_t i = 0; i < objects.size(); i++)
	{
		GraphicsData o = objects[i];
		if (o.vb < 1)
		{
			continue;
		}

		int index = i < objectRenderParams[target.pointId].size() ? i : objectRenderParams[target.pointId].size() - 1;
		ObjectRenderParams *renderParams = objectRenderParams[target.pointId][index].get();

		if (!renderParams->visible)
		{
			continue;
		}

		GLuint texId = i < texturesIDs[target.pointId].size() ? texturesIDs[target.pointId][i] :
			texturesIDs[target.pointId].empty() ? 0 : texturesIDs[target.pointId].back();
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texId);

		auto shader = renderParams->shader;

		if (renderParams->normalMapID != 0)
		{
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, renderParams->normalMapID);
		}

		ShaderSetter shaderSetter(shader, *this);

		SetUniformsForObject(*shader, i, target.pointId);

		for (int i = 0; i < BONES_COUNT; ++i)
		{
			shader->setUniform1f(PointCalcUtil::StructUniformField1("bones", i, "length"), 1, skeleton.bones[i].length);
			shader->setUniform2fv(PointCalcUtil::StructUniformField1("bones", i, "begin"), 1, skeleton.bones[i].begin.data());
			shader->setUniform2fv(PointCalcUtil::StructUniformField1("bones", i, "direction"), 1, skeleton.bones[i].direction.data());
		}

		for (int i = 0; i < BONES_COUNT; ++i)
		{
			shader->setUniform2fv(PointCalcUtil::StructUniformField1("boneTransformations", i, "joint"), 1, boneTransformations[target.pointId][i].joint.data());
			shader->setUniform2fv(PointCalcUtil::StructUniformField1("boneTransformations", i, "shift"), 1, boneTransformations[target.pointId][i].shift.data());
			shader->setUniformMatrix2fv(PointCalcUtil::StructUniformField1("boneTransformations", i, "rotation"), 1, false, boneTransformations[target.pointId][i].rotation.data());
			shader->setUniformMatrix3fv(PointCalcUtil::StructUniformField1("boneTransformations", i, "rotation3x3"), 1, false, boneTransformations[target.pointId][i].rotation3x3.data());
		}

		VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, o.vb));
		VertexAttribSetter vNormal(VertexAttribSetter::NormalAttribSetter(*shader, o.nb));
		VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*shader, o.tb));
		VertexAttribSetter vTangent(VertexAttribSetter::TangentAttribSetter(*shader, o.tangentb));
		VertexAttribSetter vBitangent(VertexAttribSetter::BitangentAttribSetter(*shader, o.bitangentb));

		VertexAttribSetter weightsCountSetter(*shader, "WeightsCount", GL_INT, 1, skeleton.weightsBuffer, 0, sizeof(Weights));
		VertexAttribSetter BonesIndexes012Setter(*shader, "BonesIndexes012", GL_INT, 3, skeleton.weightsBuffer, 4, sizeof(Weights));
		VertexAttribSetter BonesIndexes345Setter(*shader, "BonesIndexes345", GL_INT, 3, skeleton.weightsBuffer, 16, sizeof(Weights));
		VertexAttribSetter BonesWeights012Setter(*shader, "BonesWeights012", GL_FLOAT, 3, skeleton.weightsBuffer, 28, sizeof(Weights));
		VertexAttribSetter BonesWeights345Setter(*shader, "BonesWeights345", GL_FLOAT, 3, skeleton.weightsBuffer, 40, sizeof(Weights));

		glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
	}

	glPopAttrib();
}

void FxWidgetFaceMask::unload()
{
	GraphicsModel::unload();

	for (auto &IDs : texturesIDs)
	{
		IDs.clear();
	}

	skeleton.unload();
}