import open3d as o3d
import numpy as np
import random
import argparse
import os
import json


class Cube(object):
    def __init__(self, range):
        """
        Builds a cube from x, y and z ranges
        """
        self.range = range

    """
    @classmethod
    def from_points(cls, firstcorner, secondcorner):
        """ """
        Builds a cube from the bounding points
        Rectangle.from_points(Point(0, 10, -10),
                              Point(10, 20, 0)) == Rectangle((0, 10), (10, 20), (-10, 0))
        """ """

        x = (a[:, None] < b).all(-1)
        return cls(*zip(firstcorner, secondcorner))
    """

    @classmethod
    def from_voxel_size(cls, center, voxel_size):
        """
        Builds a cube from the voxel size and center of the voxel
        """
        cls.center = center
        half_center = voxel_size / 2
        x_range = (center[0] - half_center, center[0] + half_center)
        y_range = (center[1] - half_center, center[1] + half_center)
        z_range = (center[2] - half_center, center[2] + half_center)
        range = np.array(
            [[x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]]]
        )
        return cls(range)

    def contains_points(self, p):
        """
        Returns given point is in cube
        """
        less = np.repeat(self.range, repeats=[p.shape[0]], axis=0)[:, 0::2] < p
        greater = np.repeat(self.range, repeats=[p.shape[0]], axis=0)[:, 1::2] > p
        filter = np.logical_and(less.all(axis=1), greater.all(axis=1))
        return p[filter]


def get_random_color():
    return [
        round(random.uniform(0.0, 1.0), 1),
        round(random.uniform(0.0, 1.0), 1),
        round(random.uniform(0.0, 1.0), 1),
    ]


def read_point_cloud(data_root, filename, transform):
    pcd = o3d.io.read_point_cloud(os.path.join(data_root, filename))
    translation_vec = np.array(transform, dtype=np.float64)
    return pcd.translate(translation_vec)


def get_cube(pcdarray, center, voxel_size):
    cube = Cube.from_voxel_size(center, voxel_size)
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(cube.contains_points(pcdarray))
    return pcd_voxel


def compute_overlap_ratio(pcd0, pcd1, search_voxel_size):
    matching01 = get_matching_indices(pcd0, pcd1, search_voxel_size, 1)
    matching10 = get_matching_indices(pcd1, pcd0, search_voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0.points)
    overlap1 = len(matching10) / len(pcd1.points)
    return max(overlap0, overlap1)


def get_matching_indices(source, target, search_voxel_size, K=None):
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def get_valid_neigbors(i, voxels, down_array, down_array_tree, voxel_size, min_percent, max_percent):
    print(f"Voxel[{i}] - Processing")
    pairs = []
    min_overlap_ratio, running_overlap_ratio, max_overlap_ratio = 1.1, 0, -0.1
    overall_min, overall_max = 1.1, -0.1

    voxel = voxels[i]
    [k, idx1, _] = down_array_tree.search_radius_vector_3d(down_array[i], voxel_size)
    [k, idx2, _] = down_array_tree.search_radius_vector_3d(down_array[i], voxel_size / 2)
    idx1, idx2 = np.asarray(idx1), np.asarray(idx2)
    idx = np.setdiff1d(idx1, idx2)
    neighbors = idx[1:]
    print(f"Voxel[{i}] - Neighbors: {len(neighbors)}")

    for neighbor in neighbors:
        overlap_ratio = compute_overlap_ratio(voxel, voxels[neighbor], 0.05 * 4)
        overall_min = min(overall_min, overlap_ratio)
        overall_max = max(overall_max, overlap_ratio)
        if min_percent <= overlap_ratio <= max_percent:
            pairs.append((i, neighbor, overlap_ratio))
            min_overlap_ratio = min(min_overlap_ratio, overlap_ratio)
            running_overlap_ratio += overlap_ratio
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

    print(f"Voxel[{i}] - Overall Min Overlap: {overall_min:.4f}")
    print(f"Voxel[{i}] - Overall Max Overlap: {overall_max:.4f}")

    if len(pairs) != 0:
        avg_overlap_ratio = running_overlap_ratio / len(pairs)
        print(f"Voxel[{i}] - Pairs: {len(pairs)}")
        print(f"Voxel[{i}] - Min Overlap Ratio: {min_overlap_ratio:.4f}")
        print(f"Voxel[{i}] - Max Overlap Ratio: {max_overlap_ratio:.4f}")
        print(f"Voxel[{i}] - Avg Overlap Ratio: {avg_overlap_ratio:.4f}")

    return pairs, running_overlap_ratio


class Dataset:
    def __init__(self, data_root, config, dset, out_dir):
        self.data_root = data_root
        self.name = dset["name"]
        self.seqs = {seq: config["seqs"][seq] for seq in dset["seqs"]}
        self.out_dir = out_dir

    def process(self, voxel_size, min_percent, max_percent):
        for seq in self.seqs:
            seq_data = self.seqs[seq]
            filename, transform = seq_data['filename'], seq_data['transform']
            pcd = read_point_cloud(self.data_root, filename, transform)
            pcd.paint_uniform_color(get_random_color())
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            print(pcd)
            print("Minimum bound:", min_bound)
            print("Maximum bound:", max_bound)

            pcd_n = pcd.voxel_down_sample(voxel_size=0.05)
            print(pcd_n)

            downsample_voxel_size = voxel_size / 2
            downpcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

            pcd_n_array = np.asarray(pcd_n.points)
            down_array = np.asarray(downpcd.points)

            voxels = {i: get_cube(pcd_n_array, center, voxel_size) for i, center in enumerate(down_array)}
            kdtree = o3d.geometry.KDTreeFlann(downpcd)

            print("Processing neighbors")
            pairs = []
            running_overlap_ratio = 0
            for i, voxel in voxels.items():
                i_pairs, i_running_overlap_ratio = get_valid_neigbors(i, voxels, down_array, kdtree, voxel_size,
                                                                      min_percent, max_percent)
                pairs += i_pairs
                running_overlap_ratio += i_running_overlap_ratio
            avg_overlap_ratio = running_overlap_ratio / len(pairs)
            print(f"All Pairs Avg Overlap Ratio: {avg_overlap_ratio:.4f}")

            idx = self.get_valid_points(pairs)
            filenames = self.compute_filenames(idx, seq)

            pcd_array = np.asarray(pcd.points)
            for i in idx:
                cube = get_cube(pcd_array, down_array[i], voxel_size)
                filename = filenames[i]
                o3d.io.write_point_cloud(f"{filename}.ply", cube)
                np.savez(f"{filename}.npz", pcd=np.asarray(cube.points))

            pairs_filename = os.path.join(self.out_dir, f"{self.name}@seq-{seq}.txt")
            with open(pairs_filename, 'w') as f:
                for i1, i2, rat in pairs:
                    f.write(f"{i1} {i2} {rat}\n")

    def get_valid_points(self, pairs):
        idx = set()
        for i1, i2, _ in pairs:
            idx.add(i1)
            idx.add(i2)
        return idx

    def compute_filenames(self, idx, seq):
        filenames = dict()
        for i in idx:
            filenames[i] = os.path.join(self.out_dir, f"{self.name}@seq-{seq}_{str(i).zfill(3)}")
        return filenames

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), indent=4, width=1)


def get_config():
    parser = argparse.ArgumentParser(description="Point Cloud Overlapping Patch Sampler")
    parser.add_argument(
        "-d",
        "--data-root",
        type=str,
        default="data-set",
        help="Data Root Directory",
        metavar="DIR"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="out",
        help="Output Directory",
        metavar="OUT-DIR"
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="data-config.json",
        help="Dataset Configuration File",
        metavar="data-config.json"
    )
    parser.add_argument(
        "-v",
        "--voxel-size",
        type=int,
        default=5,
        help="Voxel size"
    )
    parser.add_argument(
        "--min-percent",
        type=float,
        default=0.30,
        help="Min Correspondence Percentage"
    )
    parser.add_argument(
        "--max-percent",
        type=float,
        default=0.9,
        help="Max Correspondence Percentage"
    )
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.config_file), 'r') as conf:
        config = json.load(conf)

    return args, config


def main():
    args, config = get_config()
    datasets = [Dataset(args.data_root, config, dset, args.out_dir) for dset in config["datasets"]]
    for dataset in datasets:
        print("Processing Dataset", dataset.name)
        print(dataset)
        print("=====")
        dataset.process(args.voxel_size, args.min_percent, args.max_percent)
        print("Processing finished\n")


if __name__ == "__main__":
    main()
