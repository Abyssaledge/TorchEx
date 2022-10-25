import torch
from torch import nn as nn
from torch.autograd import Function
import scatter_ext


class ScatterSumFunction(Function):

    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor):
        """connected_components function forward.
        Args:
            src (torch.Tensor): [num_total, channel]
            index (torch.Tensor, dtype:int32): [num_total]
            num_unq (int): the number of unique sets
        """
        num_unq = index[-1] + 1
        index = index.int()
        out = src.new_zeros((num_unq, src.shape[1]))
        scatter_ext.sum(src, index, out)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, g):
        # TODO
        return None, None, None


scatter_sum = ScatterSumFunction.apply
