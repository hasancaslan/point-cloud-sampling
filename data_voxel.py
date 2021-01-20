import open3d as o3d
import numpy as np
import random


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


if __name__ == "__main__":
    data_dir = "./data-set"
    N = 1000
    voxel_size = 3
    date = "13-11-23"
    seq = "02"
    pcd_path = data_dir + f"/{date}-MergedCloud-ply.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(np.asarray(pcd.points).shape)

    print("N:", N, "voxel size:", voxel_size, "path:", pcd_path)

    print("Downsampling...")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcdarray = np.asarray(pcd.points)
    downarray = np.asarray(downpcd.points)

    print("Drawing voxels...")
    voxels = [Cube.from_voxel_size(center, voxel_size) for center in downarray]

    print("Assigning voxels...")
    pcds = []
    count = 0
    for voxel in voxels:
        count += 1
        print(f"[{count}/{len(voxels)}]")
        pcd_voxel = o3d.geometry.PointCloud()
        pcd_voxel.points = o3d.utility.Vector3dVector(voxel.contains_points(pcdarray))
        pcds.append(pcd_voxel)

    train_num = int(len(pcds) * 0.8)
    test_num = int(len(pcds) * 0.1)

    print(
        "train set num:",
        train_num,
        "test set num:",
        test_num,
        "val set num:",
        len(pcds) - train_num - test_num,
    )
    print("Writing...")
    # Write point clouds
    for i in range(len(pcds)):
        if i < train_num:
            batch = i // 15
            num = i % 15
            name = "train"
        elif i < train_num + test_num:
            batch = (i - train_num) // 15
            num = (i - train_num) % 15
            name = "test"
        else:
            batch = (i - train_num - test_num) // 15
            num = (i - train_num - test_num) % 15
            name = "val"

        path_name = "./out/"
        filename = f"{name}-{date}-{str(batch).zfill(2)}@seq-{seq}"
        file_num = str(num).zfill(3)

        print(f"Writing {filename}_{file_num}")

        o3d.io.write_point_cloud(f"{path_name}{filename}_{file_num}.ply", pcds[i])
        array = np.asarray(pcds[i].points)
        np.savez(
            f"{path_name}{filename}_{file_num}.npz",
            pcd=array,
            color=np.zeros(array.shape, dtype=float),
        )

        # Write point cloud file names into text files
        f = open(f"./out/{name}-{date}-{str(batch).zfill(2)}@seq-{seq}.txt", "a+")
        f.write(f"{filename}_{file_num}.npz\n")

        f.close()

    # Change text file's format
    for i in range(18):
        if i < 14:
            split = "train"
        elif i < 16:
            split = "test"
        else:
            split = "val"

        filename = f"./out/{split}-{date}-{str(i%2).zfill(2)}@seq-{seq}.txt"
        f = open(filename, "r")

        lines = []
        for line in f.readlines():
            lines.append(line.strip())

        f = open(filename, "w+")
        for i in range(len(lines) - 1):
            for j in range(i + 1, len(lines)):
                f.write(lines[i] + " " + lines[j] + " " + "0.000000" + "\n")

        f.close()