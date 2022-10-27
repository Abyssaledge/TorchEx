import torch
from torch import nn as nn
from torch.autograd import Function
import scatter_ext


class ScatterData:
    def __init__(self, unq_inv, unq_cnts) -> None:
        assert unq_inv.ndim in (
            1, 2), f"unq_inv in ScatterData should be 1 or 2-dims, but got {unq_inv.ndim}-dims"
        assert unq_cnts.ndim in (
            1, 2), f"unq_cnts in ScatterData should be 1 or 2-dims, but got {unq_cnts.ndim}-dims"
        self.unq_inv = unq_inv.int()
        if unq_cnts.ndim == 1:
            self.unq_cnts = unq_cnts.unsqueeze(-1)
        else:
            self.unq_cnts = unq_cnts
        self.preSum = getPreSum(self._unq_inv_int)
        self.preSum32 = getPreSum32(self.unq_cnts)
        assert self.unq_inv.shape[0] == self.preSum[-1]
        self.max_cnt = self.unq_cnts.max().item()
        self.num_unq = self.preSum.shape[0] - 1


class ScatterSumFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data: ScatterData):
        """scatter_sum function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            data (class ScatterData):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
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
    def forward(ctx, feats: torch.Tensor, data: ScatterData):
        """scatter_mean function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            src (class ScatterData):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
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
    def forward(ctx, feats: torch.Tensor, data: ScatterData):
        """scatter_max function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            data (class ScatterData):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = data.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        arg = data.unq_inv.new_zeros((num_unq, channel))
        scatter_ext.max(feats, data.preSum, out, arg, data.max_cnt)
        ctx.save_for_backward(arg)
        ctx.mark_non_differentiable(arg)
        ctx.shape = feats.shape
        return out, arg

    @staticmethod
    def backward(ctx, g_out, g_arg):
        arg = ctx.saved_tensors
        g_input = g_out.new_zeros(ctx.shape)
        arg = arg.type(torch.int64)
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
        # unq_inv[-1] is the last index of unique groups, +2 means one more than number of the sets
        num_unq = unq_inv[-1] + 2
        preSum = unq_inv.new_zeros(num_unq)
        scatter_ext.getPreSum(unq_inv, preSum)

        ctx.mark_non_differentiable(preSum)

        return preSum

    @staticmethod
    def backward(ctx, g):
        return None

class GetPreSum32Function(Function):

    @staticmethod
    def forward(ctx, unq_cnts: torch.Tensor):
        """getPreSum32 function forward from unq_cnts, in which the number of each unique group is padded to 32 times.
        Args:
            unq_cnts (torch.Tensor, dtype:int32): [num_unq]
        """
        if unq_cnts.dtype != torch.int32:
            unq_cnts = unq_cnts.int()
        num_unq = unq_cnts.shape[0]
        unq_cnts32 = unq_cnts.new_zeros(num_unq)
        scatter_ext.getUnqCnts32(unq_cnts, unq_cnts32)
        preSum32 = unq_cnts.new_zeros(num_unq+1)
        preSum32[1:] = unq_cnts32.cumsum(0)
        ctx.mark_non_differentiable(preSum32)

        return preSum32

    @staticmethod
    def backward(ctx, g):
        return None

scatter_sum = ScatterSumFunction.apply
scatter_mean = ScatterMeanFunction.apply
scatter_max = ScatterMaxFunction.apply
getPreSum = GetPreSumFunction.apply
getPreSum32 = GetPreSum32Function.apply
