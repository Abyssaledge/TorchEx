import os
import torch
import numpy as np
import pickle as pkl
from torchex import dynamic_point_pool, TorchTimer
timer = TorchTimer(20)


def get_data_info(path):
    with open(path, 'rb') as fr:
        info = pkl.load(fr)
    return info

def read_point_cloud(path):
    pc = np.fromfile(path, dtype=np.float32)
    pc = pc.reshape(-1, 6)
    return pc[:, :3]


if __name__ == '__main__':
    info_path = '/mnt/truenas/scratch/lve.fan/transdet3d/data/waymo/kitti_format/waymo_infos_val_lidarframe.pkl'
    infos = get_data_info(info_path)

    with torch.no_grad():
        for i in range(100):
            pc_idx = i * 199
            pc_path = os.path.join('/mnt/truenas/scratch/lve.fan/transdet3d/data/waymo/kitti_format', infos[pc_idx]['point_cloud']['velodyne_path'])
            pc = read_point_cloud(pc_path)

            pc = torch.from_numpy(pc).float().cuda()
            boxes = torch.from_numpy(infos[pc_idx]['annos']['boxes_in_lidar']).cuda()
            boxes = torch.cat([boxes, ]*4, 0)
            if len(boxes) == 0:
                continue

            with timer.timing('Dynamic Point Pooling'):
                ext_pts_inds, roi_inds, ext_pts_info = dynamic_point_pool(
                    boxes,
                    pc,
                    [0,0,0],
                    512,
                )
