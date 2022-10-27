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


void group_fps_launcher(
    const float *points,
    const int *group_inds,
    bool *out_mask,
    int N,
    int K,
    int num_groups
);


void group_fps(
  at::Tensor points,
  at::Tensor group_inds,
  at::Tensor out_mask,
  int K,
  int num_groups
);

void group_fps(
  at::Tensor points,
  at::Tensor group_inds,
  at::Tensor out_mask,
  int K,
  int num_groups
) {

  CHECK_INPUT(group_inds);
  CHECK_INPUT(points);
  CHECK_INPUT(out_mask);
  int N = group_inds.size(0);


  int *group_inds_data = group_inds.data_ptr<int>();
  float *points_data = points.data_ptr<float>();
  bool *out_mask_data = out_mask.data_ptr<bool>();

  group_fps_launcher(
      points_data,
      group_inds_data,
      out_mask_data,
      N,
      K,
      num_groups
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_fps, "Group-wise farthest point sampling");
}
