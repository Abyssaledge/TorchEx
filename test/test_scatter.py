import torch
import numpy as np
import torch_scatter
from torchex import scatter_sum
import time
np.random.seed(0)
device = torch.device("cuda:8")
unq_inv = torch.tensor([], dtype=torch.int32, device=device)
unq_preSum = torch.zeros((501,1), dtype=torch.int32, device=device)
cumsum = 0
for group_id in range(500):
    cnt = np.random.randint(100, 1000)
    cumsum += cnt
    unq_preSum[group_id+1] = cumsum
    unq_inv = torch.cat((unq_inv, unq_inv.new_ones(cnt)*group_id), dim=0)
unq_inv.unsqueeze_(-1)
assert unq_preSum[-1,0] == unq_inv.shape[0]
feat = unq_inv.new_ones((unq_inv.shape[0],128), dtype=torch.float32)
# start = time.time()
# ans = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='sum')
# torch.cuda.synchronize()
# print(f'time cost of torch_scatter:{(time.time()-start)*1000:.4f}ms')
start = time.time()
ans = scatter_sum(feat, unq_inv)
torch.cuda.synchronize()
print(f'time cost of torch_scatter:{(time.time()-start)*1000:.4f}ms')