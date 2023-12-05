import torch
import grid_hash_ext

from torch.autograd import Function
from ipdb import set_trace

def _flatten_3dim_coors(coors, sizes):

    x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
    x_size = sizes[0]
    y_size = sizes[1]
    z_size = sizes[2]
    assert x.max() < x_size and y.max() < y_size and z.max() < z_size
    flatten_coors = x * y_size * z_size + y * z_size + z
    return flatten_coors

def _get_dim_size(coors):

    x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
    x_size = x.max().item() + 1
    y_size = y.max().item() + 1
    z_size = z.max().item() + 1
    return x_size, y_size, z_size

class _BuildHashTableFunction(Function):

    @staticmethod
    def forward(ctx, coors, values):

        # unique the coors
        table = grid_hash_ext.build_table(coors, values)
        ctx.mark_non_differentiable(table)

        return table

    @staticmethod
    def backward(ctx, g):
        return None

class _ProbeHashTableFunction(Function):

    @staticmethod
    def forward(ctx, coors, table):

        out_values = grid_hash_ext.probe_table(coors, table)
        ctx.mark_non_differentiable(out_values)

        return out_values

    @staticmethod
    def backward(ctx, g):
        return None

grid_hash_build = _BuildHashTableFunction.apply
grid_hash_probe = _ProbeHashTableFunction.apply



class GridHash:
    
    def __init__(self, coors, debug=False):

        coors = torch.unique(coors, dim=0)

        if coors.ndim == 1:
            self.dim_size = None
        else:
            assert coors.ndim == 2 and coors.size(1) == 3
            self.dim_size = _get_dim_size(coors)
            coors = _flatten_3dim_coors(coors, self.dim_size)
        

        if coors.dtype != torch.int32:
            coors = coors.int()

        # if debug:
        #     assert (coors >= 0).all() and (coors < 2 ** 31).all()

        values = torch.arange(len(coors), device=coors.device, dtype=coors.dtype)
        self.table = grid_hash_build(coors, values)
        print(len(self.table))
        self.valid_mask = self.table[:, 0] != -1
        self.valid_table = self.table[self.valid_mask]

        if debug:
            assert self.valid_mask.sum() == len(coors)
            assert self.valid_table[:, 1].max() + 1 == len(coors)
            assert len(self.valid_table[:, 1]) == len(torch.unique(self.valid_table[:, 1]))
            if not self.valid_mask.all():
                assert (self.table[~self.valid_mask, 0] == -1).all()
    
    def probe(self, coors):

        if coors.ndim == 2:
            assert coors.size(1) == 3
            coors = _flatten_3dim_coors(coors, self.dim_size)
        else:
            assert coors.ndim == 1

        if coors.dtype != torch.int32:
            coors = coors.int()
        
        values = grid_hash_probe(coors, self.table)

        return values