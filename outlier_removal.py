import argparse
import os

import open3d as o3d


VOXEL_SIZE = 0.02
UNIFORM_COLOR = [0.5, 0.5, 0.5]
RESULT_UNIFORM_COLOR = [1, 0, 0]


def radius_outlier_removal(point_cloud, min_number_points, radius):
    voxel_pcd = point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)

    voxel_pcd.paint_uniform_color(UNIFORM_COLOR)
    pcd_tree = o3d.geometry.KDTreeFlann(voxel_pcd)

    ind = []
    for index, point in enumerate(voxel_pcd.points):
        k, _, _ = pcd_tree.search_radius_vector_3d(point, radius)
        if k >= min_number_points:
            ind.append(index)

    result_cloud = voxel_pcd.select_by_index(ind, invert=True)
    result_cloud.paint_uniform_color(RESULT_UNIFORM_COLOR)
    return result_cloud


def action(args):
    """
    Parse the arguments and launch the program
    :param args: arguments (Input and Output)
    :return: None
    """
    ipc = args.ipc
    assert os.path.exists(ipc), 'Video doesnt exist'
    assert os.path.isfile(ipc), 'Video cant be a folder'

    pcd = o3d.io.read_point_cloud(ipc)
    o3d.visualization.draw_geometries([pcd])
    outlier_cloud = radius_outlier_removal(point_cloud=pcd,
                                           min_number_points=int(args.points),
                                           radius=float(args.radius))
    o3d.visualization.draw_geometries([outlier_cloud])

    o3d.io.write_point_cloud(args.opc, outlier_cloud)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Open3D")
    # Add parser output.
    parser.add_argument("-i", "--ipc", default="./open3d/cloud_bin_2.pcd")
    parser.add_argument("-p", "--points", type=int, default=16)
    parser.add_argument("-r", "--radius", type=float, default=0.05)
    parser.add_argument("-o", "--opc", default='./open3d/output.pcd')
    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        parse_arguments()
    except Exception as e:
        print("Error:", e)
