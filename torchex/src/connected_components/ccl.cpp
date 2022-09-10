#include <torch/extension.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void get_CCL(const int N, const float *const points, const int *const labels, const float thresh_dist, int *const components, const int MAXNeighbor, int mode, bool check);

void connected_components(
    at::Tensor points,
    at::Tensor labels,
    float thresh,
    at::Tensor components,
    int MAXNeighbor, int mode, bool check) {
    CHECK_INPUT(points);
    int N = points.size(0);
    const float *pts_data = points.data_ptr<float>();
    const int *labels_data = nullptr;
    if (labels.size(0) > 0)
        labels_data = labels.data_ptr<int>();
    int *comp = components.data_ptr<int>();
    get_CCL(N, pts_data, labels_data, thresh * thresh, comp, MAXNeighbor, mode, check);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &connected_components, "connected_components forward (CUDA)");
}