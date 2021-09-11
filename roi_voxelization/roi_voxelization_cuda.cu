#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> roi_voxelization_cuda_forward(
    torch::Tensor points,
    torch::Tensor boxes,
    std::tuple<int> shape,
    int max_voxels,
    bool sparse
){
    TORCH_CHECK(points.dim() == 2, "Points shape must be [N, 3], but got");
    auto N = points.size(0);
    auto num_boxes = boxes.size(0);
    auto

}

std::vector<torch::Tensor> roi_voxelization_cuda_backward(
    torch::Tensor grad_voxel,
    torch::Tensor voxel2point_inds,
    int num_points
){

}