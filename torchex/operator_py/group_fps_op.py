import torch
from torch import nn as nn
from torch.autograd import Function
import group_fps_ext
from .timer import TorchTimer
from ipdb import set_trace
timer = TorchTimer(10)

class GroupFPSFunction(Function):

    @staticmethod
    def forward(ctx, points, group_inds, num_sampling_per_group, return_points=False):
        if group_inds is None:
            group_inds = points.new_zeros(len(points), dtype=torch.int32)
        assert len(points) == len(group_inds)
        assert num_sampling_per_group > 0
        assert len(points) > 0

        group_inds, order = group_inds.sort()
        points = points[order]

        out_mask = points.new_zeros(len(points), dtype=torch.bool)

        group_fps_ext.forward(points[:, :3].contiguous(), group_inds, out_mask, num_sampling_per_group, -1) # force true for debug

        if not return_points:
            reorder_mask = torch.zeros_like(out_mask)
            reorder_mask[order] = out_mask
            output = reorder_mask
        else:
            output = points[out_mask]

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, x):

        return None, None, None

group_fps = GroupFPSFunction.apply