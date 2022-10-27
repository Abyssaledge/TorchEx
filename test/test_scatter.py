import torch
import numpy as np
import torch_scatter
from torchex import scatter_sum, scatter_max, scatter_mean, TorchTimer, ScatterMeta
from torchex.operator_py.scatter_op import get_sorted_group_inds
from ipdb import set_trace
import random

freq = 5
timer_torch_scatter = TorchTimer(freq)
timer_torchex = TorchTimer(freq)
np.random.seed(0)
random.seed(0)
# channel = 128
# num_group = 500
# cnt_min = 100
# cnt_max = 5000
# unq_inv = torch.tensor([], dtype=int, device=device)
# feat = torch.tensor([], dtype=torch.float32, device=device)
# unq_cnts = torch.zeros(num_group, dtype=int, device=device)
# for group_id in range(num_group):
#     cnt = np.random.randint(cnt_min, cnt_max)
#     unq_cnts[group_id] = cnt
#     temp_feat = torch.cat([torch.randperm(
#         cnt, dtype=torch.float32, device=device).unsqueeze_(-1)]*channel, dim=1)
#     feat = torch.cat([feat, temp_feat], dim=0)
#     unq_inv = torch.cat((unq_inv, unq_inv.new_ones(cnt)*group_id), dim=0)
# data = ScatterMeta(unq_inv, unq_cnts)


def check_method(feat, coors, mode='sum'):
    _, unq_inv, unq_cnts = torch.unique_consecutive(coors, return_inverse=True, return_counts=True, dim=0)
    meta = ScatterMeta(unq_inv, unq_cnts)
    assert mode in ['sum', 'mean', 'max']

    with timer_torch_scatter.timing(f'torch_scatter.scatter_{str(mode)}'):
        if mode == 'sum':
            ans1 = torch_scatter.scatter(
                feat, unq_inv, dim=0, reduce="sum")
        elif mode == 'mean':
            ans1 = torch_scatter.scatter_mean(feat, unq_inv, dim=0)
        else:
            ans1, arg1 = torch_scatter.scatter_max(feat, unq_inv, dim=0)

    with timer_torchex.timing(f'Torchex.scatter_{str(mode)}'):
        if mode == 'sum':
            ans2 = scatter_sum(feat, meta)
        elif mode == 'mean':
            ans2 = scatter_mean(feat, meta)
        else:
            ans2, arg2 = scatter_max(feat, meta)

    flag1 = torch.isclose(ans1, ans2).all()

    if mode == 'max':
        flag2 = (arg1 == arg2).all()
    else:
        flag2 = True
    
    if not flag2:
        print('Indices Error')
        # set_trace()
    if not flag1:
        print('Value Error')
        # set_trace()


if __name__ == '__main__':
    device = torch.device("cuda:0")
    mode = 'max'
    size = random.randint(1, 20000)
    C = random.randint(1, 1000)
    for i in range(100):
        coors_x = torch.randint(0, 10, (size,))
        coors_y = torch.randint(0, 10, (size,))
        coors_z = torch.randint(0, 10, (size,))
        coors = torch.stack([coors_x, coors_y, coors_z], dim=1).int().to(device)
        coors, order = get_sorted_group_inds(coors)

        feats = torch.rand(size, C, dtype=torch.float).to(device)

        check_method(feats, coors, mode='max')
