import os
import torch
import pickle as pkl
from torchex import chamfer_distance, TorchTimer
timer = TorchTimer(5)


def get_data_info(path):
    with open(path, 'rb') as fr:
        info = pkl.load(fr)
    return info

if __name__ == '__main__':
    info_path = '/mnt/truenas/scratch/yiming.mao/cd_test_data.pkl'
    raw_pts = get_data_info(info_path)

    a_pc = torch.from_numpy(raw_pts["pts1"]).cuda().unsqueeze(0)
    b_pc = torch.from_numpy(raw_pts["pts2"]).cuda().unsqueeze(0)
    
    dist1,dist2 = chamfer_distance(a_pc, b_pc)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

