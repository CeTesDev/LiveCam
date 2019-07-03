#include <Filters/FxDistortionAncestor.h>
#include <Graphics/GraphicsCommon.h>

class FxPinch : public FX
{
public:
	FxPinch();
	~FxPinch();

	void transformMesh(cv::Mat frame, std::vector<FXModel>& faces, Mesh3D *model) override;

private:
	void transformMeshOne(cv::Mat frame, const FXModel& fxModel, Mesh2D& imgMesh);
};