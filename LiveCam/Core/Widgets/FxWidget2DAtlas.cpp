#include <Widgets/FxWidget2DAtlas.h>
#include <Graphics/GraphicsLoader.h>

FxWidget2DAtlas::FxWidget2DAtlas()
{
}

void FxWidget2DAtlas::applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams& commonRenderParams,
	bool loadTexturesImmediately)
{
	GraphicsModel::applySuit(suit, targetIndex, commonRenderParams, loadTexturesImmediately);

	if (!visible[targetIndex])
	{
		return;
	}

	auto renderParamsID = suit.get_optional<size_t>("renderParamsID");
	if (renderParamsID)
	{
		objectRenderParams[targetIndex].clear();
		objectRenderParams[targetIndex].push_back(commonRenderParams[renderParamsID.get()]);
	}

	symmetricState[targetIndex] = suit.get<int>("symmetricState", 0);

	animationPaths[targetIndex] = suit.get<std::string>("animationAtlasPath", "");

	if (loadTexturesImmediately)
	{
		auto loadingResult = resourceManager.loadAtlas(animationPaths[targetIndex]);

		atlasTextureScale[targetIndex] = loadingResult.atlasTextureSize;
		atlasTextures[targetIndex] = loadingResult.atlasTextures;
		animationLength[targetIndex] = loadingResult.count;

		animationAtlasID[targetIndex] = loadingResult.ID;
		animationSize[targetIndex] = { loadingResult.atlasWidth, loadingResult.atlasHeight };
	}

	animationIndexes[targetIndex] = 0;
	animationState[targetIndex] = OPENING_ANIM;

	animationRotation[targetIndex] = suit.get<bool>("animationRotation", false);
	animationRolling[targetIndex] = suit.get<bool>("animationRolling", true);
	animationBackward[targetIndex] = suit.get<bool>("animationBackward", true);
	animationSpeed[targetIndex] = suit.get<float>("animationSpeed", 1);

	match3D[targetIndex] = suit.get<bool>("match3D", false);

	auto tree = suit.get_child_optional("animationOffset");
	animationOffset[targetIndex] = tree ? JSONVectorReader::readVector3f(tree.get()) : Eigen::Vector3f(0, 0, 0);

	tree = suit.get_child_optional("animationScale");
	animationScale[targetIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(1, 1);

	tree = suit.get_child_optional("animationPivot");
	animationPivot[targetIndex] = tree ? JSONVectorReader::readVector2f(tree.get()) : Eigen::Vector2f(0, 0);
}

bool FxWidget2DAtlas::load()
{
	objects.push_back(GraphicsLoader::CreateQuadModel());

	for (int targetIndex = 0; targetIndex < MAX_TO_TRACK; ++targetIndex)
	{
		auto loadingResult = resourceManager.loadAtlas(animationPaths[targetIndex]);

		atlasTextureScale[targetIndex] = loadingResult.atlasTextureSize;
		atlasTextures[targetIndex] = loadingResult.atlasTextures;
		animationLength[targetIndex] = loadingResult.count;

		animationAtlasID[targetIndex] = loadingResult.ID;
		animationSize[targetIndex] = { loadingResult.atlasWidth, loadingResult.atlasHeight };

		animationIndexes[targetIndex] = 0;
		animationState[targetIndex] = OPENING_ANIM;
	}

	return true;
}

void FxWidget2DAtlas::transform(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	double vAngle = externalRenderParams.vAngle;

	double aspect = externalRenderParams.frameWidth / static_cast<double>(externalRenderParams.frameHeight);

	double width = externalRenderParams.trackedWidth;
	double height = externalRenderParams.trackedHeight;
	double zFar = externalRenderParams.zFar;
	double zNear = externalRenderParams.zNear;

	Matrix4f projection = Matrix4f::Zero();

	projection(0, 0) = 1;
	projection(1, 1) = 1;
	projection(3, 3) = 1;

	if (match3D[face.pointId])
	{
		projection(2, 2) = (zFar + zNear) / (zNear - zFar);
		projection(2, 3) = 2 * zFar * zNear / (zNear - zFar);
	}
	else
	{
		projection(2, 2) = 2 / (zNear - zFar);
		projection(2, 3) = (zFar + zNear) / (zNear - zFar);
	}

	renderParams.projection = projection;

	auto xRawSmooth = renderParams.xCenterSmoother[face.pointId].smooth(face.xCenterRaw);
	auto yRawSmooth = renderParams.yCenterSmoother[face.pointId].smooth(face.yCenterRaw);

	auto widthRawSmooth = renderParams.widthSmoother[face.pointId].smooth(face.widthRaw);

	double rollSmooth, pitchSmooth, yawSmooth;

	pitchSmooth = renderParams.pitchSmoother[face.pointId].smooth(face.pitch);
	yawSmooth = renderParams.yawSmoother[face.pointId].smooth(face.yaw);
	rollSmooth = renderParams.rollSmoother[face.pointId].smooth(face.roll);

	double objWidth = widthRawSmooth / face.frameWidth;

	const double STD_FACE_WIDTH = 0.172;
	const double STD_FACE_DISTANCE = 0.6;
	const double STD_HEAD_LENGTH = 0.3;

	double distance = STD_FACE_WIDTH / (objWidth / STD_FACE_DISTANCE);
	depth = -distance;

	Matrix4d modelMatrix;
	modelMatrix.setIdentity();

	double xShift = ((xRawSmooth / externalRenderParams.trackedWidth) - 0.5) * 2;
	double yShift = -((yRawSmooth / externalRenderParams.trackedHeight) - 0.5) * 2;

	Affine3d rotation;
	Affine3d anti_rotation = Affine3d::Identity();

	rotation = Affine3d(AngleAxisd(-pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0)));
	rotation *= AngleAxisd(-yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
	rotation *= AngleAxisd(rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));

	if (!animationRolling[face.pointId])
	{
		anti_rotation *= AngleAxisd(-rollSmooth * M_PI / 180.0, Vector3d(0, 0, 1));
	}

	if (animationRotation[face.pointId])
	{
		renderParams.yawMatrix = Affine3d(AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0))).matrix().block<3, 3>(0, 0).cast<float>();
	}
	else
	{
		renderParams.yawMatrix = Matrix3f::Identity();
		anti_rotation *= AngleAxisd(yawSmooth * M_PI / 180.0, Vector3d(0, 1, 0));
		anti_rotation *= AngleAxisd(pitchSmooth * M_PI / 180.0, Vector3d(1, 0, 0));
	}

	bool XMirror = symmetricState[face.pointId] != 0 && -symmetricState[face.pointId] == ((0 < yawSmooth) - (yawSmooth < 0));

	modelMatrix *= Affine3d(Translation3d(Vector3d(xShift, yShift, -distance))).matrix();

	modelMatrix *= Affine3d(Scaling(1., 1. * width / height, 1.)).matrix();

	modelMatrix *= rotation.matrix();

	double Xoffset = (XMirror ? -1.0 : 1.0) * animationOffset[face.pointId][0] * objWidth;
	double Yoffset = animationOffset[face.pointId][1] * objWidth;
	double Zoffset = -animationOffset[face.pointId][2] * STD_HEAD_LENGTH;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, Zoffset))).matrix();

	modelMatrix *= anti_rotation.matrix();

	double Xscale = (XMirror ? -1.0 : 1.0) * animationScale[face.pointId][0];
	double Yscale = -animationScale[face.pointId][1] * atlasTextureScale[face.pointId][1] * animationSize[face.pointId][1] /
		atlasTextureScale[face.pointId][0] / animationSize[face.pointId][0];

	Xoffset = animationPivot[face.pointId][0] * objWidth * Xscale;
	Yoffset = -animationPivot[face.pointId][1] * objWidth * Yscale;
	modelMatrix *= Affine3d(Translation3d(Vector3d(Xoffset, Yoffset, 0))).matrix();

	modelMatrix *= Affine3d(Scaling(objWidth, objWidth, 0.5)).matrix();
	modelMatrix *= Affine3d(Scaling(Xscale, Yscale, 1.)).matrix();

	renderParams.rotationMatrix = rotation.matrix().block<3, 3>(0, 0).cast<float>();

	renderParams.modelView = modelMatrix.cast<float>();

	animate(face);
}

void FxWidget2DAtlas::draw(TrackingTarget& face, ExternalRenderParams &externalRenderParams)
{
	if (animationIndexes[face.pointId] > animationLength[face.pointId] - 2e-2)
	{
		return;
	}

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glEnable(GL_BLEND);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, animationAtlasID[face.pointId]);

	auto shader = objectRenderParams[face.pointId][0]->shader;

	ShaderSetter shaderSetter(shader, *this);

	auto textureShift = atlasTextures[face.pointId][animationIndexes[face.pointId]].shift;

	shader->setUniform2f("textureScale", atlasTextureScale[face.pointId][0], atlasTextureScale[face.pointId][1]);
	shader->setUniform2f("textureShift", textureShift[0], textureShift[1]);

	SetUniformsForObject(*shader, 0, face.pointId);

	VertexAttribSetter vPosition(VertexAttribSetter::PositionAttribSetter(*shader, objects[0].vb));
	VertexAttribSetter vTexCoord(VertexAttribSetter::TexCoordAttribSetter(*shader, objects[0].tb));

	glDrawArrays(GL_TRIANGLES, 0, 6);

	glPopAttrib();
}

void FxWidget2DAtlas::unload()
{
	GraphicsModel::unload();

	for (auto &rects : atlasTextures)
	{
		rects.clear();
	}
}