#ifndef LIVECAM_TRACKING_CONSTANTS_H
#define LIVECAM_TRACKING_CONSTANTS_H

#include <map>
#include <string>
#include <opencv2/core.hpp>

//  Global Constants for Tracking values
static const int MAX_TARGETS_COUNT = 4;
static const int MAX_TO_TRACK = MAX_TARGETS_COUNT;
static const int TOTAL_DETAIL_POINTS = 66;
static const int EXTENDED_DETAIL_POINTS = 11;
static const int TARGET_DETAIL_LIMIT = 77;
static const int TARGET_TOP_INDEX = 71;
static const int TARGET_BOTTOM_INDEX = 8;
static const int TARGET_DETAIL_MODIFIER = TOTAL_DETAIL_POINTS + 2;
static const int NUMBER_OF_PRESET_VALUES = 33;

static const int FACE_CENTER_X = 0;
static const int FACE_CENTER_Y = 1;
static const int PITCH = 2;
static const int YAW = 3;
static const int ROLL = 4;
static const int LEFT_PUPIL_X = 5;
static const int LEFT_PUPIL_Y = 6;
static const int RIGHT_PUPIL_X = 7;
static const int RIGHT_PUPIL_Y = 8;
static const int LEFT_GAZE_X = 9;
static const int LEFT_GAZE_Y = 10;
static const int RIGHT_GAZE_X = 11;
static const int RIGHT_GAZE_Y = 12;
static const int FACE_WIDTH = 13;
static const int POINT_ID = 14;
static const int POINT_TOTAL = 15;
static const int TOP_LIP_CENTER_X = 16;
static const int TOP_LIP_CENTER_Y = 17;
static const int HEAD_3D_POS_X = 18;
static const int HEAD_3D_POS_Y = 19;
static const int HEAD_3D_POS_Z = 20;
static const int HEAD_3D_POS_CC_X = 21;
static const int HEAD_3D_POS_CC_Y = 22;
static const int HEAD_3D_POS_CC_Z = 23;
static const int FACE_SCALE = 24;

static const int FACE_RECT_X = 25;
static const int FACE_RECT_Y = 26;
static const int FACE_RECT_WIDTH = 27;
static const int FACE_RECT_HEIGHT = 28;

static const int LEFT_GAZE_Z = 29;
static const int RIGHT_GAZE_Z = 30;

static const int LIP_BOTTOM_CENTER_X = 31;
static const int LIP_BOTTOM_CENTER_Y = 32;
static const int RAW_POINTS = 0;
static const int RAW_3D_POINTS = 1;

static const float CONFIDENCE_THRESHOLD = 0.3f;
static const float NMS_THRESHOLD = 0.24f;

enum class TRACKER_TYPE { 
	FACE,
	SURFACE, 
	HANDS,
	OBJECT,
	MISC
};

static const std::map<std::string, TRACKER_TYPE> TRACKER_TYPEMAP {
	{ "FACE", TRACKER_TYPE::FACE },
	{ "SURFACE", TRACKER_TYPE::SURFACE },
	{ "HANDS", TRACKER_TYPE::HANDS },
	{ "OBJECT", TRACKER_TYPE::OBJECT },
	{ "MISC", TRACKER_TYPE::MISC }
};

enum class TRACKER_LIB {
	CAFFE,
	TENSORFLOW,
	YOLO,
	MISC
};

static const std::map<std::string, TRACKER_LIB> TRACKER_LIBMAP{
	{ "CAFFE", TRACKER_LIB::CAFFE },
	{ "TENSORFLOW", TRACKER_LIB::TENSORFLOW },
	{ "YOLO", TRACKER_LIB::YOLO },
	{ "MISC", TRACKER_LIB::MISC }
};

static const cv::String CAFFE_FACE_FILE_1 = "Assets/model/caffe/deploy.prototxt";
static const cv::String CAFFE_FACE_FILE_2 = "Assets/model/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel";
static const cv::String CAFFE_KEYPOINTS_FILE_1 = "Assets/model/caffe/landmark_deploy.prototxt";
static const cv::String CAFFE_KEYPOINTS_FILE_2 = "Assets/model/caffe/VanFace.caffemodel";
static const cv::String TENSORFLOW_FACE_FILE_1 = "Assets/model/tensorflow/opencv_face_detector_uint8.pb";
static const cv::String TENSORFLOW_FACE_FILE_2 = "Assets/model/tensorflow/opencv_face_detector.pbtxt";
static const cv::String YOLO_FACE_FILE_1 = "Assets/model/yolo/tiny-yolo-azface-fddb.cfg";
static const cv::String YOLO_FACE_FILE_2 = "Assets/model/yolo/tiny-yolo-azface-fddb_82000.weights";
static const cv::String TENSORFLOW_HAND_FILE_1 = "Assets/model/tensorflow/frozen_inference_graph.pb";
static const cv::String TENSORFLOW_HAND_FILE_2 = "Assets/model/tensorflow/hand_label_map.pbtxt";
//static const cv::String TENSORFLOW_HAND_FILE_1 = "Assets/model/tensorflow/hand_inference_graph.pb";
//static const cv::String TENSORFLOW_HAND_FILE_2 = "Assets/model/tensorflow/hand_inference_graph.pbtxt";
static const cv::String CAFFE_HAND_FILE_1 = "Assets/model/caffe/openpose_pose_coco.prototxt";
static const cv::String CAFFE_HAND_FILE_2 = "Assets/model/caffe/pose_iter_440000.caffemodel";

#endif
