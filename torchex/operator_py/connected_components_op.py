import torch
from torch import nn as nn
from torch.autograd import Function

import connected_components_labeling

class ConnectedComponentsFunction(Function):

    @staticmethod
    def forward(ctx, pts, thresh_dist, max_neighbor=100, check=False):
        """connected_components function forward.
        Args:
            pts (torch.Tensor): [npoints, 3]
            thresh_dist (float): the farthest distance between two points
            max_neighbor(int): the maximum number of each point's neighbors
            check(bool): whether to check the results with bfs
        """
        if pts.device.type == 'cpu':
            pts = pts.cuda()
        components = torch.zeros(pts.shape[0], dtype=torch.int32)
        connected_components_labeling.forward(pts, thresh_dist, components, max_neighbor, check)
        

        ctx.mark_non_differentiable(components)

        return components

    @staticmethod
    def backward(ctx, components):

        return None, None, None, None

connected_components = ConnectedComponentsFunction.apply