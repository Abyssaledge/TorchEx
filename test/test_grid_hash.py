import os
import torch
from ipdb import set_trace
from torchex import GridHash
import random


if __name__ == '__main__':
    random.seed(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    for i in range(100):
        print('***********')
        all_size = random.randint(1, 500000)
        high = random.randint(10, 1024)
        coors = torch.randint(low=0, high=high, size=(all_size, 3)).int().cuda()

        unq_coors = torch.unique(coors, dim=0)

        # table = grid_hash_build(unq_coors)
        hasher = GridHash(coors, debug=True)
        out = hasher.probe(unq_coors)
        assert (out == torch.arange(len(out), device=out.device)).all()
        table = hasher.table

        valid_mask = hasher.valid_mask

        valid_values = hasher.valid_table[:, 1]

        # make sure the mapping is right

        out_valid_values = hasher.probe(table[valid_mask, 0])
        assert (out_valid_values == valid_values).all()

        # make sure the mapping is stable

        index = torch.randint(low=0, high=all_size - 1, size=(random.randint(1, 500000),) ).long().cuda()
        queries = coors

        values_1 = hasher.probe(queries)

        values_2 = hasher.probe(queries[index])
        assert (values_1[index] == values_2).all()
