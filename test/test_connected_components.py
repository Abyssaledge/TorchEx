import os
import torch
import numpy as np
import pickle as pkl
from torchex import connected_components, TorchTimer
timer = TorchTimer(20)


def get_data_info(path):
    with open(path, 'rb') as fr:
        info = pkl.load(fr)
    return info

if __name__ == '__main__':
    info_path = 'test/test_data/ccl_test_data.pkl'
    infos = get_data_info(info_path)

    with torch.no_grad():
        for i in range(100):
            pts = torch.from_numpy(infos[i]).float().cuda()
            if len(pts) == 0:
                continue

            with timer.timing('Connected Components Labeling'):
                components = connected_components(
                    pts,
                    0.5,
                    100,
                    False,
                )
