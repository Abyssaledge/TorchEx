import os
import torch
import numpy as np
import pickle as pkl
from torchex import incremental_points_mask, TorchTimer
timer = TorchTimer(20)
from ipdb import set_trace


def get_data_info(path):
    with open(path, 'rb') as fr:
        info = pkl.load(fr)
    return info

def read_point_cloud(path):
    pc = np.fromfile(path, dtype=np.float32)
    pc = pc.reshape(-1, 6)
    return pc[:, :3]

def voxelize_single(points, voxel_size, pc_range):
    device = points.device
    voxel_size = torch.tensor(voxel_size, device=device)
    pc_range = torch.tensor(pc_range, device=device)
    res_coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int() # I guess int may be faster than long type.
    # res_coors = res_coors[:, [2, 1, 0]] # to zyx order
    return res_coors


if __name__ == '__main__':
    info_path = '/mnt/truenas/scratch/lve.fan/transdet3d/data/waymo/kitti_format/waymo_infos_val_lidarframe.pkl'
    infos = get_data_info(info_path)

    data_root = '/mnt/truenas/scratch/lve.fan/transdet3d/data/waymo/kitti_format'
    base_pc_path = os.path.join(data_root, infos[0]['point_cloud']['velodyne_path'])
    inc_pc_path = os.path.join(data_root, infos[1]['point_cloud']['velodyne_path'])
    base_pc = torch.from_numpy(read_point_cloud(base_pc_path)).cuda().float()
    inc_pc = torch.from_numpy(read_point_cloud(inc_pc_path)).cuda().float()
    voxel_size = [0.25, 0.25, 0.4]
    pc_range = [-80, -80, -2, 80, 80, 4]
    shape = (640, 640, 15)


    base_coors = voxelize_single(base_pc, voxel_size, pc_range)
    inc_coors = voxelize_single(inc_pc, voxel_size, pc_range)

    base_coors_mask = (base_coors[:, 0] < shape[0]) & (base_coors[:, 1] < shape[1]) & (base_coors[:, 2] < shape[2])
    inc_coors_mask = (inc_coors[:, 0] < shape[0]) & (inc_coors[:, 1] < shape[1]) & (inc_coors[:, 2] < shape[2])

    base_coors = base_coors[base_coors_mask]
    inc_coors = inc_coors[inc_coors_mask]


    for _ in range(100):
        mask = incremental_points_mask(
            base_coors,
            inc_coors,
            shape
        )
