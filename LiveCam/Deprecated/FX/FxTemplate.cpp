#include <fx/FxTemplate.h>

FxTemplate::FxTemplate(FilterUiModel* externalInfo)
{
	loadFromJSON(boost::filesystem::path(externalInfo->getJSONsource()), externalInfo);
}

FxTemplate::~FxTemplate()
{

}

void FxTemplate::load()
{
	FxCore::load();
}

void FxTemplate::transform(FXModel& face, ExternalRenderParams &externalRenderParams)
{
	FxCore::transform(face, externalRenderParams);

	for (auto model : models)
	{
		if (model->getTypeName() != FxWidget3D::TYPE_NAME)
		{
			continue;
		}

		auto model3D = (FxWidget3D*)model.get();

		if (!model3D->backClipping)
		{
			continue;
		}

		Eigen::Vector4f minimum = { model3D->bmin[0], model3D->bmin[1], model3D->bmin[2], 1 };
		Eigen::Vector4f maximum = { model3D->bmax[0], model3D->bmax[1], model3D->bmax[2], 1 };

		Eigen::Matrix4f headAdjusting = model3D->renderParams.additionalMatrices4[0];

		Eigen::Matrix3f inversedRotation = model3D->renderParams.rotationMatrix.inverse();

		auto xRotationData = inversedRotation.block<1, 3>(0, 0).data();
		Eigen::Vector3f xRotation = { xRotationData[0], xRotationData[1], xRotationData[2] };

		auto zRotationData = model3D->renderParams.rotationMatrix.block<1, 3>(2, 0).data();
		Eigen::Vector3f zRotation = { zRotationData[0], zRotationData[1], zRotationData[2] };

		minimum = headAdjusting * minimum;
		maximum = headAdjusting * maximum;

		float Xclip = (maximum[0] - minimum[0]) / 2 * model3D->Xclip;
		float Yclip = minimum[1] + (maximum[1] - minimum[1]) * model3D->Yclip;
		float Zclip = minimum[2] + (maximum[2] - minimum[2]) * model3D->Zclip;

		for (auto &param : model3D->objectRenderParams[face.pointId])
		{
			param->additionalUniforms["XClip"] = TUniform1f(Xclip);
			param->additionalUniforms["YClip"] = TUniform1f(Yclip);
			param->additionalUniforms["ZClip"] = TUniform1f(Zclip);
			param->additionalUniforms["XRotation"] = TUniform3f(xRotation);
			param->additionalUniforms["ZRotation"] = TUniform3f(zRotation);
			param->additionalUniforms["HeadAdjustingMatrix"] = TUniform16(headAdjusting);

		}
	}
}