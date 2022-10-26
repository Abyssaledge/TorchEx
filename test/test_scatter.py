import torch
import numpy as np
import torch_scatter
from torchex import scatter_sum, scatter_max, scatter_mean, TorchTimer, ScatterData
freq = 5
timer_torch_scatter = TorchTimer(freq)
timer_torchex = TorchTimer(freq)
np.random.seed(0)
device = torch.device("cuda:0")
channel = 128
num_group = 500
cnt_min = 100
cnt_max = 5000
unq_inv = torch.tensor([], dtype=int, device=device)
feat = torch.tensor([], dtype=torch.float32, device=device)
unq_cnts = torch.zeros(num_group, dtype=int, device=device)
for group_id in range(num_group):
    cnt = np.random.randint(cnt_min, cnt_max)
    unq_cnts[group_id] = cnt
    temp_feat = torch.cat([torch.randperm(
        cnt, dtype=torch.float32, device=device).unsqueeze_(-1)]*channel, dim=1)
    feat = torch.cat([feat, temp_feat], dim=0)
    unq_inv = torch.cat((unq_inv, unq_inv.new_ones(cnt)*group_id), dim=0)
data = ScatterData(unq_inv, unq_cnts)


def check_method(times, mode='sum'):
    assert mode in ['sum', 'mean', 'max']
    for i in range(times):
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
                ans2 = scatter_sum(feat, data)
            elif mode == 'mean':
                ans2 = scatter_mean(feat, data)
            else:
                ans2, arg2 = scatter_max(feat, data)
        flag1 = bool(((ans1-ans2) < 1e-2).all().cpu())
        if mode == 'max':
            flag2 = bool((arg1 == arg2).all().cpu())
        else:
            flag2 = True
        if not flag1 or not flag2:
            print(f'check:\tans {flag1} index {flag2}')


if __name__ == '__main__':
    check_method(times=100, mode='sum')
    # check_method(times=100, mode='mean')
    # check_method(times=100, mode='max')
