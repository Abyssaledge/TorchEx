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


 void incremental_points_launcher(
  const int *unq_base_coors,
  const int *inc_coors,
  bool *out_mask,
  int N_unq,
  int N_base,
  int N_out
 );


void incremental_points_gpu(
  at::Tensor unq_base_coors,
  at::Tensor inc_coors,
  at::Tensor out_mask,
  int N_unq
) {

  CHECK_INPUT(unq_base_coors);
  CHECK_INPUT(inc_coors);
  CHECK_INPUT(out_mask);


  const int *unq_base_coors_data = unq_base_coors.data_ptr<int>();
  const int *inc_coors_data = inc_coors.data_ptr<int>();
  bool *out_mask_data = out_mask.data_ptr<bool>();
  int N_base = unq_base_coors.size(0);
  int N_out = inc_coors.size(0);

  incremental_points_launcher(
    unq_base_coors_data,
    inc_coors_data,
    out_mask_data,
    N_unq,
    N_base,
    N_out
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &incremental_points_gpu, "detect incremental points");
}
