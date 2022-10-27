import os
import torch
from ipdb import set_trace
from torchex import group_fps, TorchTimer, ingroup_inds
import random
timer = TorchTimer(10)


if __name__ == '__main__':
    random.seed(2)

    for i in range(100):
        print('***********')
        size = random.randint(10000, 100000)
        num_groups = random.randint(1, 50)
        # num_groups = 3
        K = random.randint(1, 128)
        # K =  9
        points = (torch.rand(size, 3).cuda() - 0.5) * 10
        group_inds = torch.randint(0, num_groups, (size,), dtype=torch.int32, device=points.device)
        # print(f'unique inds: {torch.unique(group_inds)}')
        out_mask = group_fps(points, group_inds, K)
        n_sampled = out_mask.sum().item()

        in_inds = ingroup_inds(group_inds.long())
        valid_mask = in_inds < K
        print(valid_mask.sum().item(), n_sampled)

    """
    corner case test
    """

    # size = 1
    # num_groups = 1
    # K = 128
    # points = (torch.rand(size, 5).cuda() - 0.5) * 10
    # group_inds = torch.randint(0, num_groups, (size,), dtype=torch.int32, device=points.device)

    # out_mask = group_fps(points, group_inds, K)