from math import ceil
import torch
from torch import nn as nn
from torch.autograd import Function
import scatter_ext
from .timer import TorchTimer
timer = TorchTimer(100)


class ScatterMeta:
    def __init__(self, unq_coors, unq_inv, unq_cnts, version=3, train=False) -> None:
        assert unq_inv.ndim in (
            1, 2), f"unq_inv in ScatterMeta should be 1 or 2-dims, but got {unq_inv.ndim}-dims"
        assert unq_cnts.ndim in (
            1, 2), f"unq_cnts in ScatterMeta should be 1 or 2-dims, but got {unq_cnts.ndim}-dims"
        self.training = train
        self.unq_inv = unq_inv.long()
        self._unq_inv_int = unq_inv.int()
        self.unq_coors = unq_coors
        if unq_cnts.ndim == 1:
            self.unq_cnts = unq_cnts.unsqueeze(-1)
        else:
            self.unq_cnts = unq_cnts
        self.num_unq = self.unq_cnts.shape[0]
        if version != 3 or self.training:
            self.preSum = getPreSum(self._unq_inv_int)
            assert self.unq_inv.shape[0] == self.preSum[-1]

        if version == 1:
            self.max_cnt = self.unq_cnts.max().item()

        if version == 2:
            self.preSum32 = getPreSum32(self.unq_cnts)
            self.Idx2Unq = self.getIdx2Unq(self.preSum32)
            self.blockDim = self.getBestBlockDim(self.preSum32[-1].item())
        
        if version == 3 and not self.training:
            self.UnqIdx, self.preSum_extend = divideUnqCnts(self.unq_cnts, max_cnt=128)
            assert(self.preSum_extend[-1] == self.unq_inv.shape[0])

    def getIdx2Unq(self, preSum32):
        Idx2Unq = preSum32.new_zeros(preSum32[-1])
        for i in range(preSum32.shape[0]-1):
            Idx2Unq[preSum32[i]:preSum32[i+1]] = i
        return Idx2Unq

    def getBestBlockDim(self, num_total32):
        feasible = [1024, 512, 256, 128]
        blockDim = 32
        util_ratio = 0.
        remainder = 1
        for temp_block in feasible:
            remainder = num_total32 % temp_block
            temp_util_ratio = num_total32 / (ceil(num_total32/temp_block)*temp_block)
            if temp_util_ratio > util_ratio:
                blockDim = temp_block
                util_ratio = temp_util_ratio
            if remainder == 0:
                break
        return blockDim


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

def divideUnqCnts(unq_cnts, max_cnt=128):
    last_idx = torch.ceil(unq_cnts/max_cnt).cumsum(dim=0, dtype=torch.int64) - 1
    remainder = torch.remainder(unq_cnts, max_cnt).int()
    remainder[remainder==0] = max_cnt
    num_unq_extend = last_idx[-1].item() + 1
    unq_cnts_extend = unq_cnts.new_ones(num_unq_extend, dtype=torch.int32) * max_cnt
    unq_cnts_extend[last_idx] = remainder
    preSum_extend = unq_cnts.new_zeros(num_unq_extend + 1, dtype=torch.int32)
    preSum_extend[1:] = unq_cnts_extend.cumsum(dim=0)
    UnqIdx = unq_cnts.new_zeros(num_unq_extend, dtype=torch.int32)
    scatter_ext.getUnqIdx(last_idx.int(), UnqIdx)
    return UnqIdx, preSum_extend


class ScatterSumFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_sum function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = meta.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        scatter_ext.sum(feats, meta.preSum, out, meta.max_cnt)
        ctx.save_for_backward(meta.unq_inv)

        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv = ctx.saved_tensors
        g_input = g[unq_inv]
        return g_input, None


class ScatterSumV2Function(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_sumV2 function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = meta.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((channel, num_unq))
        num_total32 = meta.preSum32[-1].item()
        feats_input = feats.T.contiguous()
        scatter_ext.sumV2(feats_input, meta.preSum, meta.preSum32, meta.Idx2Unq, out, num_total32, meta.blockDim)
        ctx.save_for_backward(meta.unq_inv)
        out = out.T.contiguous()

        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv = ctx.saved_tensors
        g_input = g[unq_inv]
        return g_input, None


class ScatterSumV3Function(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_sumV3 function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = meta.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        scatter_ext.sumV3(feats, meta.preSum_extend, meta.UnqIdx, out)
        ctx.save_for_backward(meta.unq_inv)

        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv = ctx.saved_tensors
        g_input = g[unq_inv]
        return g_input, None


class ScatterMeanFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_mean function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """

        sum_value = scatter_sum(feats, meta)
        out = sum_value / meta.unq_cnts
        ctx.save_for_backward(meta.unq_inv, meta.unq_cnts)
        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv, unq_cnts = ctx.saved_tensors
        g_mean = g / unq_cnts
        g_input = g_mean[unq_inv]
        return g_input, None


class ScatterMeanV3Function(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_meanV3 function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """

        sum_value = scatter_sumV3(feats, meta)
        out = sum_value / meta.unq_cnts
        ctx.save_for_backward(meta.unq_inv, meta.unq_cnts)
        return out

    @staticmethod
    def backward(ctx, g):
        unq_inv, unq_cnts = ctx.saved_tensors
        g_mean = g / unq_cnts
        g_input = g_mean[unq_inv]
        return g_input, None


class ScatterMaxFunction(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_max function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = meta.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))
        arg = meta._unq_inv_int.new_zeros((num_unq, channel))
        scatter_ext.max(feats, meta.preSum, out, arg, meta.max_cnt)
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


class ScatterMaxV3Function(Function):

    @staticmethod
    def forward(ctx, feats: torch.Tensor, meta: ScatterMeta):
        """scatter_maxV3 function forward.
        Args:
            feats (torch.Tensor, dtype:float32): [num_total, channel]
            meta (class ScatterMeta):
                unq_inv (torch.Tensor, dtype:int32): [num_total]
                unq_cnts (torch.Tensor): [num_unq] number of feats in each unique group
                preSum (torch.Tensor, dtype:int32): [num_unq+1]
                preSum32 (torch.Tensor, dtype:int32): [num_unq+1]
                max_cnt (int32): maximum number of feats in unique groups
                num_unq (int32): number of unique groups
        """
        num_unq = meta.num_unq
        channel = feats.shape[1]
        out = feats.new_zeros((num_unq, channel))-1e10
        ctx.training = meta.training
        ctx.shape = feats.shape
        arg = None
        if meta.training:
            arg = meta._unq_inv_int.new_zeros((num_unq, channel))
            scatter_ext.maxV3_train(feats, meta.preSum, out, arg)
            ctx.mark_non_differentiable(arg)
            ctx.save_for_backward(arg)
        else:
            scatter_ext.maxV3_infer(feats, meta.preSum_extend, meta.UnqIdx, out)
            ctx.mark_non_differentiable(out)
        return out, arg

    @staticmethod
    def backward(ctx, g_out, g_arg):
        if ctx.training:
            arg = ctx.saved_tensors
            g_input = g_out.new_zeros(ctx.shape)
            arg = arg.long()
            g_input.scatter_(0, arg, g_out)
            return g_input, None
        else:
            return None, None


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
scatter_sumV2 = ScatterSumV2Function.apply
scatter_sumV3 = ScatterSumV3Function.apply
scatter_mean = ScatterMeanFunction.apply
scatter_meanV3 = ScatterMeanV3Function.apply
scatter_max = ScatterMaxFunction.apply
scatter_maxV3 = ScatterMaxV3Function.apply
getPreSum = GetPreSumFunction.apply
getPreSum32 = GetPreSum32Function.apply
