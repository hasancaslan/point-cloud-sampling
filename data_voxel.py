import open3d as o3d
import numpy as np
import random
import argparse
import os
import json
from tqdm import tqdm


def get_random_color():
    return [
        round(random.uniform(0.0, 1.0), 1),
        round(random.uniform(0.0, 1.0), 1),
        round(random.uniform(0.0, 1.0), 1),
    ]


def roi_rectangle(pcdarray, minxy, size):
    maxxy = minxy + size
    xy_pts = pcdarray[:, [0, 1]]
    inidx = np.all((minxy <= xy_pts) & (xy_pts <= maxxy), axis=1)
    return pcdarray[inidx]


def get_rectangle_pcd(pcdarray, minxy, size):
    pts = roi_rectangle(pcdarray, minxy, size)
    if len(pts) == 0:
        return None
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(pts)
    return pcd_voxel


def compute_overlap_ratio(pcd0, pcd1, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, voxel_size)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, voxel_size)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def get_matching_indices(source, target, search_voxel_size):
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(point, search_voxel_size, 1)
        idx = idx[:1]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def get_valid_pairs(voxel_pairs, min_percent, matching_search_voxel_size=0.2):
    pairs = []
    min_overlap_ratio, running_overlap_ratio, max_overlap_ratio = 1.1, 0, -0.1
    overall_min, overall_max = 1.1, -0.1

    for pcd1, pcd2 in tqdm(voxel_pairs):
        overlap_ratio = compute_overlap_ratio(pcd1, pcd2, matching_search_voxel_size)
        overall_min = min(overall_min, overlap_ratio)
        overall_max = max(overall_max, overlap_ratio)
        if min_percent <= overlap_ratio:
            pairs.append((pcd1, pcd2, overlap_ratio))
            min_overlap_ratio = min(min_overlap_ratio, overlap_ratio)
            running_overlap_ratio += overlap_ratio
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

    stats = [
        ("Overall Min Overlap", overall_min),
        ("Overall Max Overlap", overall_max),
    ]

    if len(pairs) != 0:
        avg_overlap_ratio = running_overlap_ratio / len(pairs)
        stats.extend([
            ("Min Overlap Ratio", min_overlap_ratio),
            ("Max Overlap Ratio", max_overlap_ratio),
            ("Avg Overlap Ratio", avg_overlap_ratio),
        ])

    return pairs, stats


class Sequence:
    def __init__(self, data_root, name, seq_config):
        self.name = name
        self.path = os.path.join(data_root, seq_config['filename'])
        self.transform = seq_config['transform']
        self._pcd = None

    def _read_pcd(self):
        pcd = o3d.io.read_point_cloud(self.path)
        translation_vec = np.array(self.transform, dtype=np.float64)
        self._pcd = pcd.translate(translation_vec)
        print(f"Read pcd {self}:\n\t{self._pcd}")

    def _check_pcd(self):
        if self._pcd is None:
            self._read_pcd()

    def get_pcd(self):
        self._check_pcd()
        return self._pcd

    def get_pcd_array(self):
        self._check_pcd()
        return np.asarray(self._pcd.points)

    def get_bounds(self):
        self._check_pcd()
        bbox = self._pcd.get_axis_aligned_bounding_box()
        return bbox.get_min_bound(), bbox.get_max_bound()

    def __repr__(self):
        return f"Sequence \"{self.name}\" at \"{self.path}\""


class Dataset:
    def __init__(self, data_root, seq_map, dset, out_dir):
        self.data_root = data_root
        self.name = dset["name"]
        self.first_seq = seq_map[dset["first"]]
        self.second_seq = seq_map[dset["second"]]
        self.rect_size = np.array(dset["rectangle-size"])
        self.interval = np.array(dset["interval"])
        self.min_percent = dset["min-percent"]
        self.out_dir = out_dir

    def process(self):
        min_b1, max_b1 = self.first_seq.get_bounds()
        min_b2, max_b2 = self.second_seq.get_bounds()
        min_bound, max_bound = np.amin([min_b1, min_b2], axis=0), np.amax([max_b1, max_b2], axis=0)
        print("Minimum bound:", min_bound)
        print("Maximum bound:", max_bound)
        x_starts = np.arange(min_bound[0], max_bound[0] - self.rect_size[0] / 2, self.interval[0])
        y_starts = np.arange(min_bound[1], max_bound[1] - self.rect_size[1] / 2, self.interval[1])
        minxy_points = np.transpose([np.tile(x_starts, len(y_starts)), np.repeat(y_starts, len(x_starts))])

        first_pcd_points = self.first_seq.get_pcd_array()
        second_pcd_points = self.second_seq.get_pcd_array()
        voxel_pairs = []
        print(f"Reading pairs, {len(minxy_points)} candidates")
        for minxy in tqdm(minxy_points):
            voxel1 = get_rectangle_pcd(first_pcd_points, minxy, self.rect_size)
            if voxel1 is None:
                continue
            voxel2 = get_rectangle_pcd(second_pcd_points, minxy, self.rect_size)
            if voxel2 is None:
                continue
            voxel_pairs.append((voxel1, voxel2))
        print(f"Read {len(voxel_pairs)} pairs")
        print("Filtering valid pairs")
        valid_pairs, stats = get_valid_pairs(voxel_pairs, self.min_percent)
        print(f"Filtered {len(valid_pairs)} valid pairs")
        for name, value in stats:
            print(f"{name}: {value:.4f}")

        print("Writing pair files")
        filenames = []
        for i, (pcd1, pcd2, ratio) in enumerate(tqdm(valid_pairs)):
            filename1 = f"{self.name}@seq-{self.first_seq.name}_{str(i).zfill(3)}"
            filename2 = f"{self.name}@seq-{self.second_seq.name}_{str(i).zfill(3)}"
            filenames.append((f"{filename1}.npz", f"{filename2}.npz", ratio))

            filename1 = os.path.join(self.out_dir, filename1)
            o3d.io.write_point_cloud(f"{filename1}.ply", pcd1)
            np.savez(f"{filename1}.npz", pcd=np.asarray(pcd1.points))

            filename2 = os.path.join(self.out_dir, filename2)
            o3d.io.write_point_cloud(f"{filename2}.ply", pcd2)
            np.savez(f"{filename2}.npz", pcd=np.asarray(pcd2.points))

        dset_file = os.path.join(self.out_dir, f"dataset-{self.name}.txt")
        with open(dset_file, 'w') as f:
            for f1, f2, rat in filenames:
                f.write(f"{f1} {f2} {rat}\n")

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
    args = parser.parse_args()
    with open(os.path.join(args.data_root, args.config_file), 'r') as conf:
        config = json.load(conf)

    return args, config


def main():
    args, config = get_config()
    sequences = {name: Sequence(args.data_root, name, seq) for name, seq in config["seqs"].items()}
    for dset in config["datasets"]:
        dataset = Dataset(args.data_root, sequences, dset, args.out_dir)
        print("Processing Dataset", dataset.name)
        print(dataset)
        print("=====")
        dataset.process()
        print("Processing finished\n")


if __name__ == "__main__":
    main()
