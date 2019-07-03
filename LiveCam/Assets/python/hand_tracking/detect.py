from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

score_thresh = 0.27
opencv_out = 1
boxes, scores, image_np

# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def get_hands():
    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
    return boxes


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
