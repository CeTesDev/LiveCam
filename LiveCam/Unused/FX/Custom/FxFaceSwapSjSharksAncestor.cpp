#include <fx/FxSwapSjSharksAncestor.h>
#include <utils/DrawUtil.h>
#include <iostream>
#include "utils/Resolutions.h"

extern ObjectTracker tracker;

FxSwapSjSharksAncestor::FxSwapSjSharksAncestor()
{
	

}

FxSwapSjSharksAncestor::~FxSwapSjSharksAncestor()
{
}

cv::Point2i FxSwapSjSharksAncestor::getPoint(int i, const std::array<float, 66 * 2>& pts)
{
	return cv::Point2i(pts[(i - 1) * 2], pts[(i - 1) * 2 + 1]);
}

void FxSwapSjSharksAncestor::applyModule(FilterModule *module, size_t targetIndex, bool loadTExturesImmediately)
{
	QString path = QString::fromStdString(module->suits[0].get<std::string>("face", ""));

	selection(path, targetIndex);
}

void FxSwapSjSharksAncestor::selection(QString absolutePath, int index)
{
	
}

void FxSwapSjSharksAncestor::drawOnCVFrame_OLD(cv::Mat frame, FXModel& fxModel) {
	const std::array<float, 66 * 2>&  pts = fxModel.pts;
	frame_size = cv::Size(frame.cols, frame.rows);
	points_faces[fxModel.pointId][0] = getPoint(1, pts);
	points_faces[fxModel.pointId][1] = getPoint(4, pts);
	points_faces[fxModel.pointId][2] = getPoint(7, pts);
	points_faces[fxModel.pointId][3] = getPoint(9, pts);
	points_faces[fxModel.pointId][4] = getPoint(12, pts);
	points_faces[fxModel.pointId][5] = getPoint(15, pts);
	points_faces[fxModel.pointId][6] = getPoint(17, pts);
	cv::Point2i nose_length = getPoint(28, pts) - getPoint(29, pts);
	points_faces[fxModel.pointId][7] = getPoint(27, pts) + nose_length;
	points_faces[fxModel.pointId][8] = getPoint(18, pts) + nose_length;
	affine_transform_keypoints_faces[fxModel.pointId][0] = points_faces[fxModel.pointId][3];
	affine_transform_keypoints_faces[fxModel.pointId][1] = getPoint(37, pts);
	affine_transform_keypoints_faces[fxModel.pointId][2] = getPoint(46, pts);
	
	cv::Point2i box_left_corner = points_faces[fxModel.pointId][8];
	int width = PointCalcUtil::distanceBetween(points_faces[fxModel.pointId][0], points_faces[fxModel.pointId][6]);
	
	float percentToAdjust = (float) ((float)width / (float)playerFaceWidth);

	int height = PointCalcUtil::distanceBetween(box_left_corner, points_faces[fxModel.pointId][3]);// ((int)fxModel.widthRaw * 1.5) + extraSpace;

	static const int halfDim = 30;
	cv::Rect bounding_rect(box_left_corner.x - halfDim, box_left_corner.y - halfDim, width + halfDim, height + halfDim);

	int sampleWidth = 200;//;
	cv::Rect samplingRect_rect(fxModel.xCenterRaw - (sampleWidth / 2), fxModel.yCenterRaw - (sampleWidth / 2), sampleWidth, sampleWidth);
	trans_ann_to_face[fxModel.pointId] = cv::getAffineTransform(affine_transform_keypoints_ann, affine_transform_keypoints_faces[fxModel.pointId]);
	mask_face[fxModel.pointId].create(frame_size, CV_8UC1);
	
	mask_face[fxModel.pointId].setTo(cv::Scalar::all(0));
	cv::fillConvexPoly(mask_face[fxModel.pointId], points_faces[fxModel.pointId], 9, cv::Scalar(255));

	try
	{
		cv::warpAffine(mask_ann, warpped_mask_ann, trans_ann_to_face[fxModel.pointId], frame_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
	
		cv::bitwise_and(mask_face[fxModel.pointId], warpped_mask_ann, refined_face_and_ann_warpped[fxModel.pointId]);
		cv::Mat refined_masks_temp(frame_size, CV_8UC1, cv::Scalar(0));
		refined_face_and_ann_warpped[fxModel.pointId].copyTo(refined_masks_temp, refined_face_and_ann_warpped[fxModel.pointId]);
		refined_masks = refined_masks_temp;
		cv::Mat warpped_facesTemp(frame_size, CV_8UC3, cv::Scalar::all(0));
		cv::warpAffine(face_ann, warpped_face_ann, trans_ann_to_face[fxModel.pointId], frame_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		warpped_face_ann.copyTo(warpped_facesTemp, warpped_mask_ann);
		warpped_faces = warpped_facesTemp;
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	try
	{
		cv::Mat one = frame(bounding_rect);
		cv::Mat two = warpped_faces(bounding_rect);
		cv::Mat three = warpped_mask_ann(bounding_rect);
		specifiyHistogram(one, two, three);
		auto refined_mask_face = refined_masks(bounding_rect);
		featherMask(refined_mask_face);
		pasteFacesOnFrame(frame);
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	
}

void FxSwapSjSharksAncestor::featherMask(cv::Mat &refined_masks)
{
	cv::erode(refined_masks, refined_masks, getStructuringElement(cv::MORPH_RECT, feather_amount), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::blur(refined_masks, refined_masks, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);
}

inline void FxSwapSjSharksAncestor::pasteFacesOnFrame(cv::Mat frameVal)
{
	for (size_t i = 0; i < frameVal.rows; i++)
	{
		auto frame_pixel = frameVal.row(i).data;
		auto faces_pixel = warpped_faces.row(i).data;
		auto masks_pixel = refined_masks.row(i).data;

		for (size_t j = 0; j < frameVal.cols; j++)
		{
			if (*masks_pixel != 0)
			{
				*frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8;
				*(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8;
				*(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8;
			}

			frame_pixel += 3;
			faces_pixel += 3;
			masks_pixel++;
		}
	}
}

void FxSwapSjSharksAncestor::specifiyHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask)
{
	uint8_t LUT[3][256];
	int source_hist_int[3][256];
	int target_hist_int[3][256];
	float source_histogram[3][256];
	float target_histogram[3][256];

	std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
	std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_source_pixel = source_image.row(i).data;
		auto current_target_pixel = target_image.row(i).data;

		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0) {
				source_hist_int[0][*current_source_pixel]++;
				source_hist_int[1][*(current_source_pixel + 1)]++;
				source_hist_int[2][*(current_source_pixel + 2)]++;

				target_hist_int[0][*current_target_pixel]++;
				target_hist_int[1][*(current_target_pixel + 1)]++;
				target_hist_int[2][*(current_target_pixel + 2)]++;
			}
			current_source_pixel += 3;
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}
	for (size_t i = 1; i < 256; i++)
	{
		source_hist_int[0][i] += source_hist_int[0][i - 1];
		source_hist_int[1][i] += source_hist_int[1][i - 1];
		source_hist_int[2][i] += source_hist_int[2][i - 1];

		target_hist_int[0][i] += target_hist_int[0][i - 1];
		target_hist_int[1][i] += target_hist_int[1][i - 1];
		target_hist_int[2][i] += target_hist_int[2][i - 1];
	}
	for (size_t i = 0; i < 256; i++)
	{
		source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
		source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
		source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

		target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
		target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
		target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
	}

	auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
	{
		uint8_t l = 0, r = 255, m;
		while (l < r)
		{
			m = (l + r) / 2;
			if (needle > haystack[m])
				l = m + 1;
			else
				r = m - 1;
		}
		return m;
	};

	for (size_t i = 0; i < 256; i++)
	{
		LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
		LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
		LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
	}
	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_target_pixel = target_image.row(i).data;
		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0)
			{
				*current_target_pixel = LUT[0][*current_target_pixel];
				*(current_target_pixel + 1) = LUT[1][*(current_target_pixel + 1)];
				*(current_target_pixel + 2) = LUT[2][*(current_target_pixel + 2)];
			}
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}
}