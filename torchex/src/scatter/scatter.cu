#include "../utils/error.cuh"
#include "../utils/timer.cuh"
#include <fstream>
#include <sstream>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_DIM 5
#define DIVUP(m, n) ((m + n - 1) / n)
// __global__
template <typename T>
__device__ T warpReduceSum(T sum, int blockSize) {
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}
__global__ void scatter_sum(const float *const d_feats, const float *const d_preSum, float *const d_out, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    static __shared__ uint warpSum[WARP_SIZE][MAX_DIM][THREADS_PER_BLOCK / WARP_SIZE];
    float sum[MAX_DIM] = {0.0};
    int begin = -1, end = -1;
    if (unq_idx < num_unq) {
        begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
        for (int dim = 0; dim < num_dim; dim++) {
            d_out[unq_idx * num_dim + dim] = 0;
        }
    }
    int feat_idx = begin + i;
    int laneIdx = threadIdx.x % WARP_SIZE;
    int warpIdx = threadIdx.x / WARP_SIZE;
    for (int dim = 0; dim < num_dim; dim++) {
        sum[dim] = (feat_idx < end) ? d_feats[feat_idx * num_dim + dim] : 0;
        sum[dim] = warpReduceSum(sum[dim], blockDim.x);
        if (laneIdx == 0)
            warpSum[threadIdx.y][dim][warpIdx] = sum[dim];
        __syncthreads();
        sum[dim] = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpSum[threadIdx.y][dim][threadIdx.x] : 0;
        if (warpIdx == 0)
            sum[dim] = warpReduceSum(sum[dim], blockDim.x / WARP_SIZE);
        if (threadIdx.x == 0 && unq_idx < num_unq)
            atomicAdd(&d_out[unq_idx * num_dim + dim], sum[dim]);
    }
}
int main() {
    // input
    std::vector<int> unq_preSum;
    std::ifstream infile("/home/yuxue_yang/PythonWorks/SST/unq_preSum.txt");
    std::string line;
    int word;
    if (!infile) {
        printf("Cannot open test_data.txt");
        exit(1);
    }
    int N = 0, num_cols = 1;
    while (std::getline(infile, line)) {
        std::istringstream words(line);
        if (line.length() == 0) {
            continue;
        }
        for (int i = 0; i < num_cols; i++) {
            if (words >> word) {
                unq_preSum.push_back(word);
            } else {
                printf("Error for reading test_data.txt\n");
                exit(1);
            }
        }
    }
    infile.close();
    N = unq_preSum.size() - 1;
    int num_total = 344930, max_cnt = 198;
    float *d_preSum, *d_feats;
    int preSum_mem = (N + 1) * num_cols * sizeof(int);
    int cnt_mem = N * num_cols * sizeof(int);
    int feats_mem = num_total * num_cols * sizeof(float);
    CHECK_CALL(cudaMalloc(&d_preSum, preSum_mem));
    CHECK_CALL(cudaMemcpy(d_preSum, unq_preSum.data(), preSum_mem, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMalloc(&d_feats, feats_mem));
    CHECK_CALL(cudaMemset(d_feats, 1, feats_mem));
    float *d_out;
    int out_mem = N * num_cols * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_out, out_mem));
    dim3 blockSize(THREADS_PER_BLOCK, WARP_SIZE);
    dim3 gridSize(DIVUP(max_cnt, THREADS_PER_BLOCK), DIVUP(N, WARP_SIZE));
    scatter_sum<<<gridSize, blockSize>>>(d_feats, d_preSum, d_out, N, 1);
    printf("over");
    CHECK_CALL(cudaFree(d_preSum));
    CHECK_CALL(cudaFree(d_feats));
    return 0;
}