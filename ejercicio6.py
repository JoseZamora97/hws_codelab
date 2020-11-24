import argparse
import os

import cv2

from utils import video_transformation


IMG_FILENAME_FORMAT = 'image_%08d_0.png'

# This object needs to be set-up,
# this is done in the _init_ function
hog_descriptor: cv2.HOGDescriptor


def _init_():
    global hog_descriptor

    hog_descriptor = cv2.HOGDescriptor()
    hog_descriptor.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )


def process_frame(im_bgr):
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

    pedestrians, _ = hog_descriptor.detectMultiScale(im_gray, winStride=(8, 8),
                                                     padding=(32, 32), scale=1.05)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(im_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return im_bgr


def action(args):
    """
    Parse the arguments and launch the object tracking
    :param args: arguments (Input and Output)
    :return: None
    """
    innput, output = args.images, args.out

    assert innput is not None, "Input is none"
    assert os.path.exists(innput), "No such file or directory"
    assert os.path.isdir(innput), "Input can't be a directory"

    print("Starting pedestrian detection")
    video_transformation(f"{innput}/{IMG_FILENAME_FORMAT}", output, -1, process_frame)
    print("Saving video ...")


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Pedestrian Detector")
    # Add parser input.
    parser.add_argument(
        "--images", default=None, type=str,
        help="Set the video input path",
    )
    # Add parser output.
    parser.add_argument(
        "--out", default=None, type=str,
        help="Set the the processed video output path",
    )
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --images ./pedestrian_sequence --out=./media/pedestrian_result.avi
    try:
        _init_()
        parse_arguments()
    except Exception as e:
        print("Error:", e)
