import argparse
import os

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from utils import video_transformation


EAST_PRETRAINED_PATH = "./opencv-text-detection/frozen_east_text_detection.pb"
LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

east_model: cv2.dnn_Net

RESCALE_WIDTH, RESCALE_HEIGHT = 320, 320
MIN_CONFIDENCE = 0.5


def _init_():
    global east_model
    east_model = cv2.dnn.readNet(EAST_PRETRAINED_PATH)


def get_rects_with_confidences(scores, geometry):
    num_rows, num_cols = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0, x_data1, x_data2, x_data3, angles_data = [geometry[0, x, y] for x in range(5)]
        # loop over the number of columns
        for x in range(0, num_cols):
            # if our score does not have sufficient probability, ignore it
            if scores_data[x] < MIN_CONFIDENCE:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0  # Compute offset factor (x4)
            # Extract sin on cos from rotation angle
            cos, sin = np.cos(angles_data[x]), np.sin(angles_data[x])
            # Calculate height and width of bounding box
            h, w = x_data0[x] + x_data2[x], x_data1[x] + x_data3[x]
            # Compute the starting and ending coordinates for bounding box
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x, start_y = int(end_x - w), int(end_y - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences


def process_image(im_bgr):
    # Get the ratio
    r_w, r_h = im_bgr.shape[1] / float(RESCALE_WIDTH), im_bgr.shape[0] / float(RESCALE_HEIGHT)
    # resize the image and grab the new image dimensions
    im_resized = cv2.resize(im_bgr, (RESCALE_WIDTH, RESCALE_HEIGHT))

    blob = cv2.dnn.blobFromImage(im_resized, 1.0, (RESCALE_WIDTH, RESCALE_HEIGHT),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_model.setInput(blob)
    scores, geometry = east_model.forward(LAYERS)
    rects, confidences = get_rects_with_confidences(scores, geometry)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    rects = non_max_suppression(np.array(rects), probs=confidences)

    for start_x, start_y, end_x, end_y in rects:
        # Rescale the bounding boxes using the r_w and r_h factors.
        start_x, start_y = int(start_x * r_w), int(start_y * r_h)
        end_x, end_y = int(end_x * r_w), int(end_y * r_h)
        # draw the bounding box on the image
        cv2.rectangle(im_bgr, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)

    return im_bgr


def action(args):
    """
    Parse the arguments and launch the program
    :param args: arguments (Input and Output)
    :return: None
    """
    innput = args.image if args.image else args.video

    assert innput is not None, "Image/Video is none"
    assert os.path.exists(innput), 'Image/Video doesnt exist'
    assert os.path.isfile(innput), 'Image/Video cant be a folder'

    print("Starting detection ...")
    video_transformation(innput, None, 0, process_image, window_name=innput)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Text Detection")
    group = parser.add_mutually_exclusive_group()
    # Add parser input.
    group.add_argument("--video", default=None, help="Set the video input path")
    # Add parser output.
    group.add_argument("--image", default=None, help="Set image input path")
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --image opencv-text-detection/images/lebron_james.jpg
    # or
    # --video ./media/lapiz.avi
    try:
        _init_()
        parse_arguments()
    except Exception as e:
        print("Error:", e)
