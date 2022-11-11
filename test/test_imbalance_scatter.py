import torch
import numpy as np
import torch_scatter
from torchex import scatter_sum, scatter_sumV2, scatter_sumV3, scatter_max, scatter_maxV3, scatter_mean, scatter_meanV3, TorchTimer, ScatterMeta
from torchex.operator_py.scatter_op import get_sorted_group_inds
from ipdb import set_trace
import random

freq = 10
timer_torch_scatter = TorchTimer(freq)
timer_torchex = TorchTimer(freq)
np.random.seed(0)
random.seed(0)


def check_method(feat, coors, mode='sum', version=1, train=False):
    assert version in [1,2,3], "version should be in [1,2,3]"
    unq_coors, unq_inv, unq_cnts = torch.unique_consecutive(coors, return_inverse=True, return_counts=True, dim=0)
    meta = ScatterMeta(unq_coors, unq_inv, unq_cnts, version, train=train)
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
            if version == 1:
                ans2 = scatter_sum(feat, meta)
            elif version == 2:
                ans2 = scatter_sumV2(feat, meta)
            else:
                ans2 = scatter_sumV3(feat, meta)
        elif mode == 'mean':
            if version == 1:
                ans2 = scatter_mean(feat, meta)
            elif version == 2:
                raise KeyError("scatter_meanV2 has not been defined")
            else:
                ans2 = scatter_meanV3(feat, meta)
        else:
            if version == 1:
                ans2, arg2 = scatter_max(feat, meta)
            elif version == 2:
                raise KeyError("scatter_maxV2 has not been defined")
            else:
                ans2, arg2 = scatter_maxV3(feat, meta)

    flag1 = torch.isclose(ans1, ans2).all()

    if mode == 'max' and arg2 is not None:
        unq_idx, dim = torch.where(arg1 != arg2)
        idx1 = arg1[unq_idx, dim]
        idx2 = arg2[unq_idx, dim].long()
        flag2 = torch.isclose(feat[idx1, dim], feat[idx2, dim]).all()
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
    imbalance_rate = 100
    size_limits = [[1, 10],
                [10, 10**2],
                [10**2, 10**3],
                [10**3, 10**4],
                [10**4, 10**5]]
    num_group = 100
    if imbalance_rate > 1:
        print('******** Imbalance Test ********')
    else:
        print('******** Balance Test ********')
    for limit in size_limits:
        print(f'******** Group Size Limit [{limit[0]}, {limit[1]}) ********')
        version = 3
        # version: 1 means initial version, 2 means padded version, 3 means lastest version
        # C = random.randint(1, 1000)


        for C in [64, 128, 256, 512, 1024]:
            print(f'******** Test channels {C} ********')
            for i in range(10):
                coors = []
                for group_idx in range(num_group):
                    if random.random() < 1/10:
                        coors += [group_idx]*random.randint(limit[0]*imbalance_rate, limit[1]*imbalance_rate)
                    else:
                        coors += [group_idx]*random.randint(limit[0], limit[1])
                coors = torch.tensor(coors).int().to(device)
                # coors[:2000] = 0
                coors, order = get_sorted_group_inds(coors)

                feats = torch.rand(len(coors), C, dtype=torch.float).to(device)

                check_method(feats, coors, mode=mode, version=version, train=False)
