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

void roi_point_voxelization_launcher(int boxes_num, int pts_num, 
                              int max_pts_each_voxel, int out_x, int out_y,
                              int out_z,
                              int pts_feature_dim,
                              const float *rois, const float *pts,
                              int *pts_idx_of_voxels, float *pooled_features,
                              const float *extra_wlh);


int roi_point_voxelization_gpu(at::Tensor rois, at::Tensor pts,
                        at::Tensor pts_idx_of_voxels,
                        at::Tensor pooled_features,
                        std::vector<float> extra_wlh
                        );



int roi_point_voxelization_gpu(at::Tensor rois, at::Tensor pts,
                        at::Tensor pts_idx_of_voxels,
                        at::Tensor pooled_features,
                        std::vector<float> extra_wlh
                        ) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, max_pts_each_voxel, C)
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(pooled_features);

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  int pts_feature_dim = pooled_features.size(-1);


  assert((out_x < 256) && (out_y < 256) &&
         (out_z < 256));  // we encode index with 8bit

  const float *rois_data = rois.data_ptr<float>();
  const float *pts_data = pts.data_ptr<float>();
  int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  float *pooled_features_data = pooled_features.data_ptr<float>();

  const float extra_wlh_array[3] = {extra_wlh[0], extra_wlh[1], extra_wlh[2]};

  roi_point_voxelization_launcher(
      boxes_num, pts_num, max_pts_each_voxel, out_x, out_y, out_z,
      pts_feature_dim,
      rois_data, pts_data, 
      pts_idx_of_voxels_data, pooled_features_data, extra_wlh_array);

// void roi_point_voxelization_launcher(int boxes_num, int pts_num, 
//                               int max_pts_each_voxel, int out_x, int out_y,
//                               int out_z,
//                               int pts_feature_dim,
//                               const float *rois, const float *pts,
//                               int *pts_idx_of_voxels, float *pooled_features, 
//                               const float *extra_wlh) {
  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_point_voxelization_gpu, "roiaware pool3d forward (CUDA)");
}
