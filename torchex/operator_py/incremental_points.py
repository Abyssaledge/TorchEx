import torch
from torch import nn as nn
from torch.autograd import Function

import find_incremental_points
from ipdb import set_trace
from .timer import TorchTimer
timer = TorchTimer(5)


class _FindIncPointsFunction(Function):

    @staticmethod
    def forward(ctx, base_coors, inc_coors, spatial_size, base_hashing_size=100000):

        xs, ys, zs = spatial_size
        assert 0 < xs * ys * zs < 2147483647
        base_coors = base_coors[:, 0] * (ys * zs) + base_coors[:, 1] * zs + base_coors[:, 2]
        inc_coors = inc_coors[:, 0] * (ys * zs) + inc_coors[:, 1] * zs + inc_coors[:, 2]

        out_mask = base_coors.new_zeros(len(inc_coors), dtype=torch.bool)

        find_incremental_points.forward(base_coors, inc_coors, out_mask, base_hashing_size)

        ctx.mark_non_differentiable(out_mask)

        return out_mask

    @staticmethod
    def backward(ctx, g1):

        return None, None, None, None

incremental_points_mask = _FindIncPointsFunction.apply