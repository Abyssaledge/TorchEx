import torch
from torch.autograd import Function
import codec
import numpy as np

class MaskEncoder(Function):

    @staticmethod
    def forward(ctx, mask: torch.Tensor):
        """scatter_sum function forward.
        Args:
            mask (torch.Tensor, dtype:bool): [total,] The mask of points
        """
        total = mask.shape[0]
        num_code = int(np.ceil(total / 64))
        code = mask.new_zeros(num_code, dtype=torch.int64)
        codec.encode(mask, code, total)
        ctx.mark_non_differentiable(code)
        return code

    @staticmethod
    def backward(ctx, g):

        return None

class MaskDecoder(Function):

    @staticmethod
    def forward(ctx, code: torch.Tensor, total: int):
        """scatter_sum function forward.
        Args:
            mask (torch.Tensor, dtype:int64): [np.ceil(total/64),] The code of points mask
            total (int): The number of points
        """
        mask = code.new_zeros(total, dtype=bool)
        if code.is_cuda:
            codec.decode(code, mask, total)
        else:
            codec.decode_cpu(code, mask, total)
        ctx.mark_non_differentiable(mask)
        return mask

    @staticmethod
    def backward(ctx, g):

        return None

mask_encoder = MaskEncoder.apply
mask_decoder = MaskDecoder.apply