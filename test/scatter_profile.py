import torch
import numpy as np
import scatter_ext
import torch_scatter
from torchex import scatter_sum, scatter_sumV2, scatter_sumV3, scatter_max, scatter_maxV3, scatter_mean, scatter_meanV3, TorchTimer, ScatterMeta
from torchex.operator_py.scatter_op import get_sorted_group_inds
from ipdb import set_trace
import random
import time

class Bandwidth(object):
    class NamedTimer(object):
        def __init__(self, name, print_freq):
            self.name = name
            self.time_cost = 0
            self.begin_time = 0
            self.exe_counter = 0
            self.print_freq = print_freq
            self.mem = 0
            self.bandwidth = 0

        def reset(self):
            self.time_cost = 0
            self.exe_counter = 0
            self.bandwidth = 0

        def __enter__(self):
            if self.print_freq > 0:
                torch.cuda.synchronize()
                self.begin_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.print_freq == -1:
                return
            torch.cuda.synchronize()
            self.time_cost += time.time() - self.begin_time
            self.bandwidth += self.mem / self.time_cost
            self.exe_counter += 1
            if self.exe_counter % self.print_freq == 0:
                print('Average time cost of {}: {:.4}ms'. format(self.name, self.time_cost / self.exe_counter * 1000))
                print('Average bandwidth of {}: {:.4}GB/s'. format(self.name, self.bandwidth / self.exe_counter))
                self.reset()

    def __init__(self, print_freq = 10):
        self.timer_dict = {}
        self.print_freq = print_freq
    
    def timing(self, name, freq = None, mem = 0):
        print_freq = freq if freq is not None else self.print_freq
        if name not in self.timer_dict:
            self.timer_dict[name] = Bandwidth.NamedTimer(name, print_freq)
        self.timer_dict[name].mem = mem
        return self.timer_dict[name]

freq = 5
timer_torch_scatter = Bandwidth(freq)
timer_torchex = Bandwidth(freq)
np.random.seed(0)
random.seed(0)


def check_method(feat, coors, mode='sum', version=1, train=False):
    assert version in [1,2,3], "version should be in [1,2,3]"
    unq_coors, unq_inv, unq_cnts = torch.unique_consecutive(coors, return_inverse=True, return_counts=True, dim=0)
    meta = ScatterMeta(unq_coors, unq_inv, unq_cnts, version, train=train)
    assert mode in ['sum', 'mean', 'max']

    num_unq = meta.num_unq
    channel = feat.shape[1]
    out = feat.new_zeros((num_unq, channel))-1e10
    if mode == 'max' and meta.training:
        arg = meta._unq_inv_int.new_zeros((num_unq, channel))

    mem = 0
    if mode == 'sum':
        mem = feat.numel() + meta.preSum_extend.numel() * 2 + meta.UnqIdx.numel() + out.numel()
        mem = mem * 4 / (10 ** 9)
    elif mode == 'max':
        if meta.training:
            mem = feat.numel() + meta.preSum.numel() * 2 + arg.numel() + out.numel()
            mem = mem * 4 / (10 ** 9)
        else:
            mem = feat.numel() + meta.preSum_extend.numel() * 2 + meta.UnqIdx.numel() + out.numel()
            mem = mem * 4 / (10 ** 9)
    with timer_torchex.timing(f'Torchex.scatter_{mode}', mem=mem):
        if mode == 'sum':
            scatter_ext.sumV3(feat, meta.preSum_extend, meta.UnqIdx, out)
        elif mode == 'mean':
            raise KeyError("No need for scatter_mean")
        else:
            if meta.training:
                scatter_ext.maxV3_train(feat, meta.preSum, out, arg)
            else:
                scatter_ext.maxV3_infer(feat, meta.preSum_extend, meta.UnqIdx, out)
    
    mem = feat.numel() + unq_inv.numel() + out.numel()
    mem = mem * 4 / (10 ** 9)
    with timer_torch_scatter.timing(f'torch_scatter.scatter_{str(mode)}', mem=mem):
        if mode == 'sum':
            ans1 = torch_scatter.scatter(
                feat, unq_inv, dim=0, reduce="sum")
        elif mode == 'mean':
            ans1 = torch_scatter.scatter_mean(feat, unq_inv, dim=0)
        else:
            ans1, arg1 = torch_scatter.scatter_max(feat, unq_inv, dim=0)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    mode = 'max'
    size = random.randint(100000, 500000)
    version = 3
    # version: 1 means initial version, 2 means padded version, 3 means lastest version
    # C = random.randint(1, 1000)

    for C in [64, 1000]:
        print(f'******** Test channels {C} ********')
        for i in range(10):
            coors_x = torch.randint(0, 2, (size,))
            coors_y = torch.randint(0, 2, (size,))
            coors_z = torch.randint(0, 2, (size,))
            coors = torch.stack([coors_x, coors_y, coors_z], dim=1).int().to(device)
            coors[:2000] = 0
            coors, order = get_sorted_group_inds(coors)
            feats = torch.rand(size, C, dtype=torch.float).to(device)

            check_method(feats, coors, mode, version, train=False)


            