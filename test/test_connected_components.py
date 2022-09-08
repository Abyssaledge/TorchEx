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
    team_num = 5    # 将team_num的数据合并
    with torch.no_grad():
        for i in range(0,100,team_num):
            pts = torch.concat(raw_pts[i:i+team_num], dim=0).cuda()
            labels = torch.concat(raw_labels[i:i+team_num], dim=0).cuda()
            if len(pts) == 0:
                continue

            with timer.timing('Connected Components Labeling'):
                components = connected_components(
                    pts,
                    labels, # 不带标签则用None
                    0.5,
                    75,
                    3,      # mode 2表示前两维，3表示三维
                    False,
                )
                print(f'cuda计算连通域数量为{components.max().item()+1}')
