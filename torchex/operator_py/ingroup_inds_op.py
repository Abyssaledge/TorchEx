import torch
import ingroup_indices
from torch.autograd import Function
class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):

        out_inds = torch.zeros_like(group_inds) - 1

        ingroup_indices.forward(group_inds, out_inds)

        ctx.mark_non_differentiable(out_inds)

        return out_inds

    @staticmethod
    def backward(ctx, g):

        return None

ingroup_inds = IngroupIndicesFunction.apply