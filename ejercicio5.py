import argparse
import os

import cv2

from utils import video_transformation

PRETRAINED_MODELS_PATH = "./eye_face"
EYE_MODEL_NAME = "haarcascade_eye.xml"
FRONTAL_FACE_MODEL_NAME = "haarcascade_frontalface_default.xml"

# These two objects need to be set-up,
# this is done in the _init_ function
face_cascade: cv2.CascadeClassifier
eye_cascade: cv2.CascadeClassifier


def _init_():
    """
    This method initializes the Cascade Classifiers
    for safely load the pretrained models xml files.

    :return: None
    """
    global face_cascade, eye_cascade

    face_path = os.path.join(os.path.abspath(PRETRAINED_MODELS_PATH),
                             FRONTAL_FACE_MODEL_NAME)
    eye_path = os.path.join(os.path.abspath(PRETRAINED_MODELS_PATH),
                            EYE_MODEL_NAME)

    assert os.path.exists(face_path), f'{FRONTAL_FACE_MODEL_NAME} not found'
    assert os.path.exists(eye_path), f'{EYE_MODEL_NAME} not found'

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)


def process_frame(im_bgr):
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in face_cascade.detectMultiScale(
            im_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
    ):
        cv2.rectangle(im_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray, roi_color = im_gray[y:y + h, x:x + w], im_bgr[y:y + h, x:x + w]

        for (ex, ey, ew, eh) in eye_cascade.detectMultiScale(
                roi_gray
        ):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return im_bgr


def action(args):
    """
    Parse the arguments and launch the object tracking
    :param args: arguments (Input and Output)
    :return: None
    """
    innput, output = args.video, args.out

    assert innput is not None, "Input is none"
    assert os.path.exists(innput), "No such file or directory"
    assert os.path.isfile(innput), "Input can't be a directory"

    if output is None:
        folders = innput.split("/")
        folders[-1] = "proc_" + folders[-1]
        output = "/".join(folders)

    print("Starting detection of frontal face and eyes")
    video_transformation(innput, output, 0, process_frame)
    print("Saving video ...")


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="EyeFaceDetector")
    # Add parser input.
    parser.add_argument(
        "-v",
        "--video",
        default=None,
        type=str,
        help="Set the video input path",
    )
    # Add parser output.
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        type=str,
        help="Set the the processed video output path",
    )
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --video ./media/Avengers.mp4 --out ./media/avengers_result.avi
    try:
        _init_()
        parse_arguments()
    except Exception as e:
        print("Error:", e)
