#include <torch/extension.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void get_CCL(const int N, const float *const points, const float thresh_dist, int *const components, const int MAXNeighbor, bool check);

void connected_components(at::Tensor points, float thresh, at::Tensor components, int MAXNeighbor, bool check) {
    CHECK_INPUT(points);
    int N = points.size(0);
    const float *pts_data = points.data_ptr<float>();
    // auto components = torch::zeros(N, torch::dtype(torch::kInt32));
    int *comp = components.data_ptr<int>();
    get_CCL(N, pts_data, thresh, comp, MAXNeighbor, check);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &connected_components, "connected_components forward (CUDA)");
}