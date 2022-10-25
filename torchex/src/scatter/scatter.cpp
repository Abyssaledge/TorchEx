#include <torch/extension.h>
#include <torch/serialize/tensor.h>


#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void scatter_sum_launcher(const float *const feats, const int *const unq_inv, float *const out, int num_total, int channel, int num_unq);

void scatter_sum_gpu(
    at::Tensor feats,
    at::Tensor unq_inv,
    at::Tensor out) {
    CHECK_INPUT(feats);
    CHECK_INPUT(unq_inv);
    CHECK_INPUT(out);
    int num_total = feats.size(0), channel = feats.size(1);
    int num_unq = out.size(0);
    const float *feats_data = feats.data_ptr<float>();
    const int *unq_inv_data = unq_inv.data_ptr<int>();
    float *out_data = out.data_ptr<float>();
    scatter_sum_launcher(feats_data, unq_inv_data, out_data, num_total, channel, num_unq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", &scatter_sum_gpu, "scatter_sum (CUDA)");
}