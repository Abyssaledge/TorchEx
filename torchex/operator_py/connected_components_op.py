import torch
from torch import nn as nn
from torch.autograd import Function
import connected_components_labeling

class ConnectedComponentsFunction(Function):

    @staticmethod
    def forward(ctx, pts, labels, thresh_dist, max_neighbor=100, mode=3, check=False):
        """connected_components function forward.
        Args:
            pts (torch.Tensor): [npoints, 3]
            labels (torch.Tensor): [npoints] or None
            thresh_dist (float): the farthest distance between two points
            max_neighbor(int): the maximum number of each point's neighbors
            mode(int): the number of dimensions for distance calculation(2 or 3)
            check(bool): whether to check the results with bfs
        """
        assert mode in [2,3], 'The mode must be 2 or 3\n'
        if labels is None:
            labels = pts.new_zeros(len(pts), dtype=torch.int32)
        components = pts.new_zeros(pts.shape[0], dtype=torch.int32)
        connected_components_labeling.forward(pts, labels, thresh_dist, components, max_neighbor, mode, check) # force true for debug

        ctx.mark_non_differentiable(components)

        return components

    @staticmethod
    def backward(ctx, components):

        return None, None, None, None, None, None

connected_components = ConnectedComponentsFunction.apply