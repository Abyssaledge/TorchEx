#include <torch/extension.h>

#include <vector>
#include <tuple>

// CUDA forward declarations

std::vector<torch::Tensor> roi_voxelization_cuda_forward(
    torch::Tensor points,
    torch::Tensor bbox,
    std::tuple<int> shape,
    int max_voxels,
    bool sparse
);

std::vector<torch::Tensor> roi_voxelization_cuda_backward(
    torch::Tensor grad_voxel,
    torch::Tensor voxel2point_inds,
    int num_points
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> roi_voxelization_forward(
    torch::Tensor points,
    torch::Tensor feats,
    torch::Tensor box,
    std::tuple<int> shape,
    int max_voxels,
    bool sparse
) {
  CHECK_INPUT(points);
  CHECK_INPUT(feats);
  CHECK_INPUT(box);
  auto N = points.size(0);
  auto num_features = feats.size(0);
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3 && N > 0, "Points shape should be [N, 3], but got ", points.sizes());
  TORCH_CHECK(feats.dim() == 2 && feats.size(1) > 0, && num_features > 0, "Point feature shape should be [N, C], but got ", feats.sizes());
  TORCH_CHECK(num_features == N, "Point feature shape should be [N, C], but got ", N, " v.s. ", num_features );
  TORCH_CHECK(box.dim() == 2 && box.size(0) > 0 && box.size(1) == 7, "Expect box of shape [N, 7], but got ", box.sizes());
  TORCH_CHECK(shape.size == 2 || shape.size() == 3, );

  return roi_voxelization_cuda_forward(points, bbox, shape, max_voxels, sparse);
}

std::vector<torch::Tensor> roi_voxelization_backward(
    torch::Tensor grad_voxel,
    torch::Tensor voxel2point_inds,
    int num_points
) {
  CHECK_INPUT(grad_voxel);
  CHECK_INPUT(voxel2point_inds);

  return roi_voxelization_cuda_backward(
    torch::Tensor grad_voxel,
    torch::Tensor voxel2point_inds,
    int num_points
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_voxelization_forward, "forward (CUDA)");
  m.def("backward", &roi_voxelization_backward, "backward (CUDA)");
}