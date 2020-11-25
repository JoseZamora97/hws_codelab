import argparse
import collections
import os

import cv2
import numpy as np

from utils import video_transformation

# Default values
MIN_H, MAX_H = (29, 94)
MIN_S, MAX_S = (43, 255)
MIN_V, MAX_V = (126, 255)

OUT_CODEC = 'MJPG'

NOISE_FOCUS_SIZE = (3, 3)
ERODE_ITERATIONS = 2
DILATION_ITERATIONS = 2
CONTOUR_AREA_THRESHOLD = 350

POSITIONS_BUFFER_LEN = 7


def get_mask_from_hsv(h, s, v):
    """
    Create the mask to follow the object
    :param h: h channel
    :param s: s channel
    :param v: v channel
    :return: np.array with the 3 channels
    """
    mask_h = (h > MIN_H) & (h < MAX_H)
    mask_s = (s > MIN_S) & (s < MAX_S)
    mask_v = (v > MIN_V) & (v < MAX_V)

    return mask_h & mask_s & mask_v


def find_contours(img):
    ret, thresh = cv2.threshold(img, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def print_min_circle_and_centroid(img, cnt):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    m = cv2.moments(cnt)
    centroid = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    cv2.circle(img, centroid, 5, (0, 255, 0), 10)

    return centroid


def noise_suppression(img):
    kernel_noise = np.ones(NOISE_FOCUS_SIZE, np.uint8)
    # Erode operations.
    erode_img = cv2.erode(img, kernel_noise, iterations=ERODE_ITERATIONS)
    # Dilation operation.
    return cv2.dilate(erode_img, kernel_noise, iterations=DILATION_ITERATIONS)


def process_frame(image_bgr, buffer):
    """
    Process frame.
    :param image_bgr: HSV format image array.
    :param buffer: buffer of previous images.
    :return: image_blending of processed frame
    """
    im_bgr_filtered = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    im_hsv = cv2.cvtColor(im_bgr_filtered, cv2.COLOR_BGR2HSV)

    # Get the mask from the array
    hsv_mask = get_mask_from_hsv(
        h=im_hsv[:, :, 0],
        s=im_hsv[:, :, 1],
        v=im_hsv[:, :, 2]
    )
    hsv_mask = (hsv_mask * 255).astype(np.uint8)
    _, contours, hierarchy = find_contours(noise_suppression(hsv_mask))

    x = y = -1
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
            x, y = print_min_circle_and_centroid(im_bgr_filtered, cnt)

    if x != -1 and y != -1:
        buffer.append((int(x), int(y)))

    for i in range(len(buffer)):
        if i > 1:
            cv2.line(im_bgr_filtered, buffer[i - 1], buffer[i], (255, 0, 0), 2*i)

    return im_bgr_filtered


def action(args):
    """
    Parse the arguments and launch the object tracking
    :param args: arguments (Input and Output)
    :return: None
    """
    innput, output = args.video, args.output
    min_values, max_values = args.min_values, args.max_values

    assert innput is not None, "Input is none"
    assert os.path.exists(innput), "No such file or directory"

    assert all(x > 0 for x in min_values), "All min values have to be greater than 0"
    assert all(x < 255 for x in max_values), "All max values have to be lower than 255"

    global MIN_H, MIN_S, MIN_V
    global MAX_H, MAX_S, MAX_V

    MIN_H, MIN_S, MIN_V = min_values
    MAX_H, MAX_S, MAX_V = max_values

    if output is None:
        folders = innput.split("/")
        folders[-1] = "proc_" + folders[-1]
        output = "/".join(folders)

    print("Starting object tracking")

    # Create a buffer with POSITIONS_BUFFER_LEN slots remove "maxlen"
    # param for tracking the full object movement.
    buffer = collections.deque(maxlen=POSITIONS_BUFFER_LEN)
    video_transformation(innput, output, 10, process_frame, (buffer, ))
    print("Saving video ...")


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Visual Tracking")
    # Add parser input.
    parser.add_argument("--video", default=None, required=True, help="Set the video input path")
    # Add parser output.
    parser.add_argument("--output", default=None, required=True, help="Set the the processed video output path")
    # Add parser min values
    parser.add_argument("--min_values", nargs=3, type=int, help="Set HSV min values", required=True)
    # Add parser max values
    parser.add_argument("--max_values", nargs=3, type=int, help="Set HSV max values", required=True)
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --video ./media/lapiz.avi --min_values=29 43 126 --max_values=88 255 255
    parse_arguments()
