import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

INTERVAL_MIN = 0
INTERVAL_MAX = 500
INTERVAL_SKIP = 100


def analyse_percentage_for_ranges(inference: np.array, gt: np.array):
    intervals = [(i, i + INTERVAL_SKIP)
                 for i in range(INTERVAL_MIN, INTERVAL_MAX, INTERVAL_SKIP)] + [(INTERVAL_MAX, np.inf)]

    subtraction = np.abs(inference - gt)
    nan_amount = np.isnan(subtraction).sum()
    mat_diffs = np.array([(subtraction > i) & (subtraction < j) for i, j in intervals]).astype(int)
    mat_diffs = np.sum(mat_diffs, axis=1)

    result = np.insert(mat_diffs, 0, nan_amount) / inference.size * 100

    print(result)


def plot_n_save(path_inference, path_ground_truth, output_folder):
    np_inf = np.loadtxt(path_inference, dtype=str, delimiter=",")
    np_gt = np.loadtxt(path_ground_truth, dtype=str, delimiter=",")

    np_inf[np_inf == '-'] = np_gt[np_gt == '-'] = np.NaN
    np_inf, np_gt = np_inf[1:].astype(float), np_gt[1:].astype(float)

    analyse_percentage_for_ranges(np_inf[:, 1], np_gt[:, 1])


def action(args):
    # Check the Inference path
    assert os.path.exists(args.inference), "Inference path doesn't exist"
    assert os.path.isfile(args.inference), "Inference path must be a file"
    # Check the Inference path
    assert os.path.exists(args.groundtruth), "Ground truth path doesn't exist"
    assert os.path.isfile(args.groundtruth), "Ground truth path must be a file"

    assert args.output_graphs is not None, "A folder must be specified"

    plot_n_save(args.inference, args.groundtruth, args.output_graphs)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Compute Stats")

    # Add parser factor.
    parser.add_argument(
        "--inference", default=None, required=True,
        help="Set the Inference input file (.csv)"
    )
    # Add parser input.
    parser.add_argument(
        "--groundtruth", default=None, required=True,
        help="Set the Ground Truth input file (.csv)",
    )
    # Add parser output.
    parser.add_argument(
        "--output_graphs", default=None, required=True,
        help="Set the Output Graphs directory",
    )
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    # --inference ./results/detection.csv
    # --groundtruth ./results/groundtruth.csv
    # --output_graphs ./results/output_stats
    parse_arguments()
