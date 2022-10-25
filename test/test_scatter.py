import torch
import numpy as np
import torch_scatter
from torchex import scatter_sum, TorchTimer
import time
freq = 5
timer_torch_scatter = TorchTimer(freq)
timer_torchex = TorchTimer(freq)
np.random.seed(0)
device = torch.device("cuda:0")
channel = 128
num_group = 500
unq_inv = torch.tensor([], dtype=int, device=device)
cumsum = 0
for group_id in range(num_group):
    cnt = np.random.randint(100, 1000)
    unq_inv = torch.cat((unq_inv, unq_inv.new_ones(cnt)*group_id), dim=0)
unq_inv.unsqueeze_(-1)
feat = unq_inv.new_ones((unq_inv.shape[0], channel), dtype=torch.float32)
# start = time.time()
# ans1 = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='sum')
# torch.cuda.synchronize()
# print(f'time cost of torch_scatter:\t\t{(time.time()-start)*1000:2.4f}ms')
# start = time.time()
# ans2 = scatter_sum(feat, unq_inv)
# torch.cuda.synchronize()
# print(f'time cost of scatter_sum(Torchex):\t{(time.time()-start)*1000:2.4f}ms')
# print(f'check:\t{torch.all((ans1-ans2) < 1e-3)}')

for i in range(100):
    with timer_torch_scatter.timing('torch_scatter'):
        ans1 = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='sum')
    with timer_torchex.timing('Torchex'):
        ans2 = scatter_sum(feat, unq_inv)
    if (i+1)%freq == 0:
        flag = bool(((ans1-ans2)<1e-2).all().cpu())
        print(f'check:\t{flag}')
