// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// std::vector<at::Tensor> weighted_point2voxel(
//   at::Tensor pts, at::Tensor pts_feature,
//   float tau,
//   int max_points,
//   std::vector<float> voxel_size,
//   std::vector<float> pc_range
// );


std::vector<at::Tensor> weighted_point2voxel_launcher(
  at::Tensor pts, at::Tensor pts_feature,
  float tau,
  int max_points,
  std::vector<float> voxel_size,
  std::vector<float> enlarged_voxel_size,
  std::vector<float> pc_range
  );


std::vector<at::Tensor> weighted_point2voxel(
  at::Tensor pts, at::Tensor pts_feature,
  float tau,
  int max_points,
  std::vector<float> voxel_size,
  std::vector<float> enlarged_voxel_size,
  std::vector<float> pc_range
  ) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params pts_feature: (npoints, C)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, C)
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(pts);
  CHECK_INPUT(pts_feature);
  assert(pts.size(0) == pts_feature.size(0));

  float x1 = pc_range[0];
  float y1 = pc_range[1];
  float z1 = pc_range[2];
  float x2 = pc_range[3];
  float y2 = pc_range[4];
  float z2 = pc_range[5];
  assert((x2 > x1) && (y2 > y1) && (z2 > z1));

  float v1 = voxel_size[0];
  float v2 = voxel_size[1];
  float v3 = voxel_size[2];
  assert((v1 > 0) && (v2 > 0) && (v3 > 0));

  float ev1 = elarged_voxel_size[0];
  float ev2 = elarged_voxel_size[1];
  float ev3 = elarged_voxel_size[2];
  assert((ev1 >= v1) && (ev2 > v2) && (ev3 > v3));

  std::vector<at::Tensor> output = weighted_point2voxel_launcher(pts, pts_feature, tau, max_points, voxel_size, enlarged_voxel_size, pc_range);

  return output;
}

int weighted_point2voxel_backward(at::Tensor pts_idx_of_voxels,
                                 at::Tensor argmax, at::Tensor grad_out,
                                 at::Tensor grad_in, int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(argmax);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_in);

  int boxes_num = pts_idx_of_voxels.size(0);
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int channels = grad_out.size(4);

  const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  const int *argmax_data = argmax.data_ptr<int>();
  const float *grad_out_data = grad_out.data_ptr<float>();
  float *grad_in_data = grad_in.data_ptr<float>();

  roiaware_pool3d_backward_launcher(boxes_num, out_x, out_y, out_z, channels,
                                    max_pts_each_voxel, pts_idx_of_voxels_data,
                                    argmax_data, grad_out_data, grad_in_data,
                                    pool_method);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &weighted_point2voxel, "weighted_point2voxel forward (CUDA)");
  m.def("backward", &weighted_point2voxel_backward,
        "weighted_point2voxel backward (CUDA)");
}
