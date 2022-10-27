import torch
from torch import nn as nn
from torch.autograd import Function
import scatter_ext


class ScatterMeta:
    def __init__(self, unq_coors, unq_inv, unq_cnts) -> None:
        assert unq_inv.ndim in (
            1, 2), f"unq_inv in ScatterMeta should be 1 or 2-dims, but got {unq_inv.ndim}-dims"
        assert unq_cnts.ndim in (
            1, 2), f"unq_cnts in ScatterMeta should be 1 or 2-dims, but got {unq_cnts.ndim}-dims"
        self.unq_inv = unq_inv.long()
        self._unq_inv_int = unq_inv.int()
        self.unq_coors = unq_coors
        if unq_cnts.ndim == 1:
            self.unq_cnts = unq_cnts.unsqueeze(-1)
        else:
            self.unq_cnts = unq_cnts
        self.preSum = getPreSum(self._unq_inv_int)
        assert self.unq_inv.shape[0] == self.preSum[-1]
        self.max_cnt = self.unq_cnts.max().item()
        self.num_unq = self.preSum.shape[0] - 1
    
def get_sorted_group_inds(inds):
    """
    simple way for fast develope
    """
    _inds = inds.clone()
    if inds.ndim == 1:
        dims = 1
    else:
        assert inds.ndim == 2
        min_ind = inds.min().item()
        if min_ind < 0:
            inds = inds - min_ind
        dims = inds.size(1)
    
    if dims == 1:
        return _inds.sort()

    maxs = inds.max(0)[0] + 1

    if dims == 2:
        flat_inds = inds[:, 0] * maxs[1] + inds[:, 1]
        _, order = flat_inds.sort()
        return _inds[order], order

    if dims == 3:
        flat_inds = inds[:, 0] * maxs[1] * maxs[2] + inds[:, 1] * maxs[2] + inds[:, 2]
        _, order = flat_inds.sort()
        return _inds[order], order

    raise NotImplementedError


class ScatterSumFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data: ScatterMeta):
        """scatter_sum function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            data (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique sets
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique sets
                num_unq (int32): number of unique sets
        """
        num_unq = data.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        scatter_ext.sum(feats, data.preSum, out, data.max_cnt)
        ctx.save_for_backward(data.unq_inv)

        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv = ctx.saved_tensors
        g_input = g[unq_inv]
        return g_input, None


class ScatterMeanFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data: ScatterMeta):
        """scatter_mean function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            src (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique sets
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique sets
                num_unq (int32): number of unique sets
        """

        sum_value = scatter_sum(feats, data)
        out = sum_value / data.unq_cnts
        ctx.save_for_backward(data.unq_inv, data.unq_cnts)
        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv, unq_cnts = ctx.saved_tensors
        g_mean = g / unq_cnts
        g_input = g_mean[unq_inv]
        return g_input, None


class ScatterMaxFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data: ScatterMeta):
        """scatter_max function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            data (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique sets
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique sets
                num_unq (int32): number of unique sets
        """
        num_unq = data.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        arg = data._unq_inv_int.new_zeros((num_unq, channel))
        scatter_ext.max(feats, data.preSum, out, arg, data.max_cnt)
        ctx.save_for_backward(arg)
        ctx.mark_non_differentiable(arg)
        ctx.shape = feats.shape
        return out, arg

    @staticmethod
    def backward(ctx, g_out, g_arg):
        arg = ctx.saved_tensors
        g_input = g_out.new_zeros(ctx.shape)
        arg = arg.long()
        g_input.scatter_(0, arg, g_out)
        return g_input, None


class GetPreSumFunction(Function):

    @staticmethod
    def forward(ctx, unq_inv: torch.Tensor):
        """getPreSum function forward from unq_inv.
        Args:
            unq_inv (torch.Tensor, dtype:int32): [num_total]
        """
        if unq_inv.dtype != torch.int32:
            unq_inv = unq_inv.int()
        # unq_inv[-1] is the last index of unique sets, +2 means one more than number of the sets
        num_unq = unq_inv[-1] + 2
        preSum = unq_inv.new_zeros(num_unq)
        scatter_ext.getPreSum(unq_inv, preSum)

        ctx.mark_non_differentiable(preSum)

        return preSum

    @staticmethod
    def backward(ctx, g):
        return None


scatter_sum = ScatterSumFunction.apply
scatter_mean = ScatterMeanFunction.apply
scatter_max = ScatterMaxFunction.apply
getPreSum = GetPreSumFunction.apply
