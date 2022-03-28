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

std::vector<at::Tensor> dynamic_point_pool_launcher(
  int boxes_num,
  int pts_num, 
  int max_num_pts_per_box,
  const float *rois,
  const float *pts,
  const float *extra_wlh
  );


std::vector<at::Tensor> dynamic_point_pool_gpu(
  at::Tensor rois,
  at::Tensor pts,
  std::vector<float> extra_wlh,
  int max_num_pts_per_box
);


std::vector<at::Tensor> dynamic_point_pool_gpu(
  at::Tensor rois,
  at::Tensor pts,
  std::vector<float> extra_wlh,
  int max_num_pts_per_box
) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);

  const float *rois_data = rois.data_ptr<float>();
  const float *pts_data = pts.data_ptr<float>();

  const float extra_wlh_array[3] = {extra_wlh[0], extra_wlh[1], extra_wlh[2]};

  auto out_vector = dynamic_point_pool_launcher(
    boxes_num,
    pts_num, 
    max_num_pts_per_box,
    rois_data,
    pts_data,
    extra_wlh_array
  );

  return out_vector;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dynamic_point_pool_gpu, "dynamic_point_pool_gpu forward (CUDA)");
}
