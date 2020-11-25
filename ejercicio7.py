import argparse
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

MATCHES_COLOR = None  # (0, 255, 0)
POINTS_COLOR = None  # (255, 0, 0)

MATCHES_THRESHOLD_TO_SHOW = 20
KNN_DISTANCE_PERCENTAGE = 0.75
FLANN_INDEX_KD_TREE = 1
FLANN_INDEX_LSH = 6

EXTENSIONS = ('.jpg', '.png', '.JPG', '.PNG')

DEFAULT_DETECTOR = "orb"
DETECTORS = {'orb': cv2.ORB_create, 'sift': cv2.SIFT_create}

DEFAULT_MATCHER = "flann"
MATCHER = {'bfm': cv2.BFMatcher, 'flann': cv2.FlannBasedMatcher}

MATCHERS_CONFIG = {
    "bfm": {
        'orb': {"normType": cv2.NORM_HAMMING, "crossCheck": True},
        'sift': {}
    },
    "flann": {
        'sift': {
            "indexParams": dict(algorithm=FLANN_INDEX_KD_TREE, trees=5),
            "searchParams": dict(checks=50)
        },
        'orb': {
            "indexParams": dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=12,  # 12
                key_size=20,  # 20
                multi_probe_level=1),
            "searchParams": {}
        }
    }
}


def plot_match(match, detector, matcher, output):
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

    ax2.imshow(match.get('image'))
    ax2.set_title('Matches')

    ax1.imshow(cv2.imread(match.get('path')))
    ax1.set_title('Image matched')

    fig.suptitle(f"{detector.upper()} + {matcher.upper()}", fontsize=16)

    plt.subplots_adjust(left=0.5, right=1.)
    plt.tight_layout()

    if output:
        plt.savefig(output)

    plt.show()


def search_candidates(search, covers, detector, matcher):
    im_search = cv2.imread(search, cv2.IMREAD_GRAYSCALE)
    im_search_kp, im_search_des = detector.detectAndCompute(im_search, None)

    im_candidates = []

    for cover in tqdm(covers, desc='Searching...', position=0):
        im_target = cv2.imread(cover, cv2.IMREAD_GRAYSCALE)
        im_target_kp, im_target_des = detector.detectAndCompute(im_target, None)

        if not im_target_kp or len(im_target_des) == 0:
            continue

        draw_params = {"matchColor": MATCHES_COLOR, "singlePointColor": POINTS_COLOR,
                       "flags": cv2.DrawMatchesFlags_DEFAULT}

        if type(detector) == cv2.ORB:
            draw_function = cv2.drawMatches
            matches = matcher.match(im_search_des, im_target_des)
            matches = sorted(matches, key=lambda x: x.distance)

        elif type(detector) == cv2.SIFT:
            draw_function = cv2.drawMatchesKnn
            matches = matcher.knnMatch(im_search_des, im_target_des, k=2)
            matches = [[m] for m, n in sorted(matches, key=lambda x: x[0].distance)
                       if m.distance < KNN_DISTANCE_PERCENTAGE * n.distance]
        else:
            raise RuntimeError("Error: Invalid detector")

        if len(matches) > MATCHES_THRESHOLD_TO_SHOW:
            im_composite = draw_function(
                im_search, im_search_kp, im_target, im_target_kp,
                matches, None, **draw_params)

            im_candidates.append(
                {"image": im_composite, "matches": len(matches), "path": cover}
            )

    return im_candidates


def action(args):
    """
    Parse the arguments and launch the program
    :param args: arguments (Input and Output)
    :return: None
    """
    query, covers = args.query, args.covers

    assert os.path.exists(query), 'Query image doesnt exist'
    assert os.path.isfile(query), 'Query image cant be a folder'

    assert os.path.exists(covers), 'Cover database folder doesnt exist'
    assert os.path.isdir(covers), 'Cover database folder cant be a file'

    if args.output:
        assert os.path.exists(args.output), 'Output doesnt exist'
        assert os.path.isdir(covers), 'Output cant be a file'

    detector = DETECTORS[args.detector]()
    matcher_args = MATCHERS_CONFIG.get(args.matcher).get(args.detector)
    matcher = MATCHER[args.matcher](**matcher_args)

    queue_covers = map(lambda x: x.as_posix(),
                       filter(lambda x: x.suffix in EXTENSIONS,
                              Path(args.covers).rglob('*.*')))

    ini = time.time()
    matches = search_candidates(query, queue_covers, detector, matcher)
    end = time.time()

    match = sorted(matches, key=lambda x: -x.get('matches'))[0]
    output = args.output if not args.output else \
        f"{args.output}/{args.detector}_{args.matcher}_{round(end - ini, 4)}.png"

    plot_match(match, args.detector, args.matcher, output=output)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Visual Tracking CV")
    # Add parser output.
    parser.add_argument("--query", required=True, help="Set image path to query for")
    parser.add_argument("--covers", required=True, help="Set covers database path")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, choices=list(DETECTORS.keys()),
                        help="OpenCV object detector type")
    parser.add_argument("--matcher", default=DEFAULT_MATCHER, choices=list(MATCHER.keys()),
                        help="OpenCV object matcher type")
    parser.add_argument("--output", default=None, help="output folder")
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # –query ./cover_The_Hobbit –covers ./my_media_database/
    # try:
    parse_arguments()
    # except Exception as e:
    #     print("Error:", e)
