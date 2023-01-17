#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void encoder_launcher(const bool *mask, unsigned long long *code, int total);

void decoder_launcher(const unsigned long long *code, bool *mask, int total);

void decoder_cpu_launcher(const unsigned long long *code, bool *mask, int total);

void encoder_gpu(
    at::Tensor mask,
    at::Tensor code,
    int total) {
    CHECK_INPUT(mask);
    CHECK_INPUT(code);
    const bool *mask_data = mask.data_ptr<bool>();
    unsigned long long *code_data = (unsigned long long*)code.data_ptr();
    encoder_launcher(mask_data, code_data, total);
}

void decoder_gpu(
    at::Tensor code,
    at::Tensor mask,
    int total) {
    CHECK_INPUT(mask);
    CHECK_INPUT(code);
    bool *mask_data = mask.data_ptr<bool>();
    const unsigned long long *code_data = (unsigned long long*)code.data_ptr();
    decoder_launcher(code_data, mask_data, total);
}

void decoder_cpu(
    at::Tensor code,
    at::Tensor mask,
    int total) {
    CHECK_CONTIGUOUS(mask);
    CHECK_CONTIGUOUS(code);
    bool *mask_data = mask.data_ptr<bool>();
    const unsigned long long *code_data = (unsigned long long*)code.data_ptr();
    decoder_cpu_launcher(code_data, mask_data, total);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode", &encoder_gpu, "encode points mask (CUDA)");
    m.def("decode", &decoder_gpu, "decode points mask (CUDA)");
    m.def("decode_cpu", &decoder_cpu, "decode points mask (CPU)");
}