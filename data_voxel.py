import open3d as o3d
import numpy as np
import random
from pyntcloud import PyntCloud


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


cube = Cube.from_voxel_size((5, 15, -5), 10)
points = np.array([[3, 15, -8], [11, 15, -8]])

cube.contains_points(points)

if __name__ == "__main__":
    data_dir = "./data-set"
    N = 1000
    voxel_size = 3
    pcd_path = data_dir + "/13-11-23-MergedCloud-ply.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)

    print("N:", N, "voxel size:", voxel_size, "path:", pcd_path)

    print("Downsampling...")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcdarray = np.asarray(pcd.points)
    downarray = np.asarray(downpcd.points)

    print("Drawing voxels...")
    voxels = [Cube.from_voxel_size(center, voxel_size) for center in downarray]

    print("Assigning voxels...")
    pcds = []
    for voxel in voxels:
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

        filename = (
            f"./out/{name}-12_12_14-{str(batch).zfill(2)}@seq-01_{str(num).zfill(3)}"
        )
        o3d.io.write_point_cloud(f"{filename}.ply", pcds[i])
        cloud = PyntCloud.from_file(f"{filename}.ply")
        cloud.to_file(f"{filename}.npz")

        f = open(f"./out/{name}-12_12_14-{str(batch).zfill(2)}.txt", "a+")
        f.write(
            f"{name}-12_12_14-{str(batch).zfill(2)}@seq-01_{str(num).zfill(3)}.npz\n"
        )
        f.close()
