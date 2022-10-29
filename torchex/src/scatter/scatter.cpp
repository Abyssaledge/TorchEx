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

void scatter_sumV2_launcher(const float *const feats, const int *const preSum, const int *const preSum32, const int *const Idx2Unq,
                            float *const out, int num_total, int num_total32, int num_unq, int channel, int blockDim_x);

void scatter_sumV3_launcher(const float *const feats, const int *const preSum, float *const out,
                            int channel, int num_unq);

void scatter_max_launcher(const float *const feats, const int *const preSum, float *const out, int *const arg,
                          int channel, int num_unq, int max_cnt);

void scatter_maxV3_launcher(const float *const feats, const int *const preSum, float *const out, int *const arg,
                          int channel, int num_unq);

void getPreSum_launcher(const int *const unq_inv, int *const preSum, int num_total);

void getUnqCnts32_launcher(const int *const unq_cnts, int *const unq_cnts32, int num_unq);

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

void scatter_sumV2_gpu(
    at::Tensor feats,
    at::Tensor preSum,
    at::Tensor preSum32,
    at::Tensor Idx2Unq,
    at::Tensor out,
    int num_total32,
    int blockDim_x) {
    CHECK_INPUT(feats);
    CHECK_INPUT(preSum);
    CHECK_INPUT(preSum32);
    CHECK_INPUT(Idx2Unq);
    CHECK_INPUT(out);
    int channel = feats.size(0);
    int num_total = feats.size(1);
    int num_unq = preSum.size(0) - 1;
    const float *feats_data = feats.data_ptr<float>();
    const int *preSum_data = preSum.data_ptr<int>();
    const int *preSum32_data = preSum32.data_ptr<int>();
    const int *Idx2Unq_data = Idx2Unq.data_ptr<int>();
    float *out_data = out.data_ptr<float>();
    scatter_sumV2_launcher(feats_data, preSum_data, preSum32_data, Idx2Unq_data, out_data, num_total, num_total32, num_unq, channel, blockDim_x);
}

void scatter_sumV3_gpu(
    at::Tensor feats,
    at::Tensor preSum,
    at::Tensor out) {
    CHECK_INPUT(feats);
    CHECK_INPUT(preSum);
    CHECK_INPUT(out);
    int channel = feats.size(1);
    int num_unq = out.size(0);
    const float *feats_data = feats.data_ptr<float>();
    const int *preSum_data = preSum.data_ptr<int>();
    float *out_data = out.data_ptr<float>();
    scatter_sumV3_launcher(feats_data, preSum_data, out_data, channel, num_unq);
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

void scatter_maxV3_gpu(
    at::Tensor feats,
    at::Tensor preSum,
    at::Tensor out,
    at::Tensor arg) {
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
    scatter_maxV3_launcher(feats_data, preSum_data, out_data, arg_data, channel, num_unq);
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

void getUnqCnts32_gpu(
    at::Tensor unq_cnts,
    at::Tensor unq_cnts32) {
    CHECK_INPUT(unq_cnts);
    CHECK_INPUT(unq_cnts32);
    int num_unq = unq_cnts.size(0);
    const int *unq_cnts_data = unq_cnts.data_ptr<int>();
    int *unq_cnts32_data = unq_cnts32.data_ptr<int>();
    getUnqCnts32_launcher(unq_cnts_data, unq_cnts32_data, num_unq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", &scatter_sum_gpu, "scatter_sum (CUDA)");
    m.def("sumV2", &scatter_sumV2_gpu, "scatter_sumV2 (CUDA)");
    m.def("sumV3", &scatter_sumV3_gpu, "scatter_sumV3 (CUDA)");
    m.def("max", &scatter_max_gpu, "scatter_max (CUDA)");
    m.def("maxV3", &scatter_maxV3_gpu, "scatter_maxV3 (CUDA)");
    m.def("getPreSum", &getPreSum_gpu, "get preSum from unq_inv (CUDA)");
    m.def("getUnqCnts32", &getUnqCnts32_gpu, "get unq_cnts32 from unq_cnts (CUDA)");
}