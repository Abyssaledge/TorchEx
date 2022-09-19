import os
import torch
import numpy as np
import pickle as pkl
from torchex import connected_components, TorchTimer
timer = TorchTimer(5)


def get_data_info(path):
    with open(path, 'rb') as fr:
        info = pkl.load(fr)
    return info

if __name__ == '__main__':
    info_path = 'test/test_data/ccl_test_data.pkl'
    raw_pts = get_data_info(info_path)
    raw_pts = [torch.from_numpy(raw_pts[i]).float() for i in range(100)]
    raw_labels = [torch.ones(raw_pts[i].shape[0], dtype=torch.int32)*i for i in range(100)]
    team_num = 5   
    with torch.no_grad():
        for i in range(0,100,team_num):
            pts = torch.cat(raw_pts[i:i+team_num], dim=0).cuda()
            labels = torch.cat(raw_labels[i:i+team_num], dim=0).cuda()
            if len(pts) == 0:
                continue

            with timer.timing('Connected Components Labeling'):
                components = connected_components(
                    pts,
                    labels,
                    0.5,
                    75,
                    2, # 2 for x-y distance; 3 for x-y-z distance
                    False,
                )
                # print(f'Number of cc in cuda: {components.max().item()+1}')
