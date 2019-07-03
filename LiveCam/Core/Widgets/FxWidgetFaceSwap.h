#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <Common/DrawUtil.h>
#include <Common/Resolutions.h>
#include <Graphics/GraphicsModel.h>
#include <FX.h>

class FxWidgetFaceSwap : public GraphicsModel
{
protected:
	std::array<Eigen::Vector2f, MAX_TO_TRACK> textureSize;

public:
	FxWidgetFaceSwap();

	static const std::string TYPE_NAME;

	std::array<std::string, MAX_TO_TRACK> faceSwapTextures;
	std::array<GLuint, MAX_TO_TRACK> faceSwapTexturesIDs;
	std::array<std::vector<GLuint>, MAX_TO_TRACK> texturesIDs;
	std::array<std::vector<std::string>, MAX_TO_TRACK> texturesPaths;

	std::string getTypeName() override;

	void loadFromJSON(boost::property_tree::ptree& modelRecord) override;

	void applySuit(boost::property_tree::ptree& suit, size_t targetIndex, TCommonRenderParams &commonRenderParams, bool loadTexturesImmediately) override;

	bool load() override;

	void unload() override;

	void selection(QString absolutePath, int index);

	void transform(TrackingTarget& target, ExternalRenderParams &externalRenderParams) override;
	void transformMesh(cv::Mat frame, std::vector<TrackingTarget>& targets, Mesh3D *mesh) override;
	void draw(TrackingTarget& target, ExternalRenderParams &externalRenderParams) override;

private:
	cv::Mat img1;

	cv::Mat leftDisplay;
	
	QString currentPath;
	void featherMask(cv::Mat &refined_masks);
	void pasteFacesOnFrame(cv::Mat frameVal);
	void specifiyHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask);

	cv::Point2f  affine_transform_keypoints_player[3], affine_transform_keypoints_ann[3];

	cv::Mat warpped_face_ann, warpped_face_bob;

	cv::Point2i points_player[9], points_ann[9];

	cv::Point2i getPoint(int i, const std::array<float, TARGET_DETAIL_LIMIT * 2>& pts);

	cv::Point2f affine_transform_keypoints_faces[MAX_TO_TRACK][3];
	
	cv::Mat trans_ann_to_face[MAX_TO_TRACK];
	cv::Mat mask_face[MAX_TO_TRACK];
	cv::Mat refined_face_and_ann_warpped[MAX_TO_TRACK];

	cv::Mat warpped_mask_ann, warpped_mask_bob;
	cv::Mat warpped_faces;

	cv::Mat refined_masks;
	cv::Point2i points_faces[MAX_TO_TRACK][9];
	cv::Mat mask_ann, mask_bob, mask_player;
	cv::Mat face_ann, face_bob;

	cv::Size player_size;
	cv::Size frame_size;

	cv::Size feather_amount;

	int playerFaceWidth;

};