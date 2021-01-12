import argparse
import os

import cv2

from utils import proc_video


def process_frame(im_bgr, tracker):
    success, box = tracker.update(im_bgr)
    if success:
        x, y, w, h = [int(i) for i in box]
        cv2.rectangle(im_bgr, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    return im_bgr


def action(args):
    """
    Parse the arguments and launch the program
    :param args: arguments (Input and Output)
    :return: None
    """
    innput = args.image

    assert innput is not None, "Video is none"
    assert os.path.exists(innput), 'Video doesnt exist'
    assert os.path.isfile(innput), 'Video cant be a folder'

    tracker = TRACKERS[args.tracker]()

    video_capture = cv2.VideoCapture(innput)
    _, ini_frame = video_capture.read()
    bbox = cv2.selectROI(f"Tracking {os.path.basename(innput)}", ini_frame, fromCenter=False,
                         showCrosshair=False)
    tracker.init(ini_frame, bbox)

    print("Starting tracking ...")
    proc_video(None, None, 0, process_frame, args=[tracker, ],
               window_name=f"Tracking {os.path.basename(innput)}", video_capture=video_capture)


TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Visual Tracking CV")
    # Add parser output.
    parser.add_argument("--image", required=True, help="Set video input path")
    parser.add_argument("--tracker", type=str, default="csrt",
                        choices=list(TRACKERS.keys()),
                        help="OpenCV object tracker type")
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --image ./media/bolt.avi
    try:
        parse_arguments()
    except Exception as e:
        print("Error:", e)
