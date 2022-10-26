#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void scatter_sum_launcher(const float *const feats, const int *const preSum, float *const out,
                          int channel, int num_unq, int max_cnt);

void scatter_max_launcher(const float *const feats, const int *const preSum, float *const out, int *const arg,
                          int channel, int num_unq, int max_cnt);

void getPreSum_launcher(const int *const unq_inv, int *const preSum, int num_total);

void scatter_sum_gpu(
    at::Tensor feats,
    at::Tensor preSum,
    at::Tensor out,
    int max_cnt) {
    CHECK_INPUT(feats);
    CHECK_INPUT(preSum);
    CHECK_INPUT(out);
    int channel = feats.size(1);
    int num_unq = out.size(0);
    const float *feats_data = feats.data_ptr<float>();
    const int *preSum_data = preSum.data_ptr<int>();
    float *out_data = out.data_ptr<float>();
    scatter_sum_launcher(feats_data, preSum_data, out_data, channel, num_unq, max_cnt);
}

void scatter_max_gpu(
    at::Tensor feats,
    at::Tensor preSum,
    at::Tensor out,
    at::Tensor arg,
    int max_cnt) {
    CHECK_INPUT(feats);
    CHECK_INPUT(preSum);
    CHECK_INPUT(out);
    CHECK_INPUT(arg);
    int channel = feats.size(1);
    int num_unq = out.size(0);
    const float *feats_data = feats.data_ptr<float>();
    const int *preSum_data = preSum.data_ptr<int>();
    float *out_data = out.data_ptr<float>();
    int *arg_data = arg.data_ptr<int>();
    scatter_max_launcher(feats_data, preSum_data, out_data, arg_data, channel, num_unq, max_cnt);
}

void getPreSum_gpu(
    at::Tensor unq_inv,
    at::Tensor preSum) {
    CHECK_INPUT(unq_inv);
    CHECK_INPUT(preSum);
    int num_total = unq_inv.size(0);
    const int *unq_inv_data = unq_inv.data_ptr<int>();
    int *preSum_data = preSum.data_ptr<int>();
    getPreSum_launcher(unq_inv_data, preSum_data, num_total);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", &scatter_sum_gpu, "scatter_sum (CUDA)");
    m.def("max", &scatter_max_gpu, "scatter_max (CUDA)");
    m.def("getPreSum", &getPreSum_gpu, "get preSum from unq_inv (CUDA)");
}