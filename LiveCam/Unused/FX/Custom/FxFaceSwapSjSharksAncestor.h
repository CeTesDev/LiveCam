#pragma once
#include <fx\FX.h>

class FxSwapSjSharksAncestor : public FX
{
public:
	FxSwapSjSharksAncestor();
	~FxSwapSjSharksAncestor();

	void applyModule(FilterModule *module, size_t targetIndex, bool loadTExturesImmediately) override;

	void drawOnCVFrame_OLD(cv::Mat frame, FXModel& fxModel) override;

	void selection(QString absolutePath, int index);
private:
	cv::Mat img1;

	cv::Mat leftDisplay;
	QString currentPath;
	void featherMask(cv::Mat &refined_masks);
	void pasteFacesOnFrame(cv::Mat frameVal);
	void specifiyHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask);

	cv::Point2i getPoint(int i, const std::array<float, 66 * 2>& pts);

	cv::Point2f  affine_transform_keypoints_player[3], affine_transform_keypoints_ann[3];

	cv::Point2f affine_transform_keypoints_faces[ObjectTracker::MAX_TO_TRACK][3];
	cv::Mat warpped_face_ann, warpped_face_bob;

	cv::Point2i points_player[9], points_ann[9];

	cv::Point2i points_faces[ObjectTracker::MAX_TO_TRACK][9];
	cv::Mat trans_ann_to_face[ObjectTracker::MAX_TO_TRACK];
	cv::Mat mask_face[ObjectTracker::MAX_TO_TRACK];
	cv::Mat normal_face[ObjectTracker::MAX_TO_TRACK];
	cv::Mat refined_face_and_ann_warpped[ObjectTracker::MAX_TO_TRACK];

	cv::Mat trans_ann_to_bob, trans_bob_to_ann;
	cv::Mat mask_ann, mask_bob, mask_player;
	cv::Mat warpped_mask_ann, warpped_mask_bob;
	cv::Mat refined_masks;
	cv::Mat face_ann, face_bob;
	cv::Mat warpped_faces;

	cv::Mat small_frame;

	cv::Size player_size;
	cv::Size frame_size;

	cv::Size feather_amount;

	int playerFaceWidth;
};