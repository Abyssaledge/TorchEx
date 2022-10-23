#include "../utils/error.cuh"
#include "../utils/timer.cuh"
#include <assert.h>
#include <fstream>
#include <sstream>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
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

__global__ void scatter_sum(const float *const d_feats, const int *const d_preSum, float *const d_out, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    static __shared__ uint warpSum[THREADS_PER_BLOCK][THREADS_PER_BLOCK / WARP_SIZE];
    float sum = 0;
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
        sum = (feat_idx < end) ? d_feats[feat_idx * num_dim + dim] : 0;
        sum = warpReduceSum(sum, blockDim.x);
        if (laneIdx == 0)
            warpSum[threadIdx.y][warpIdx] = sum;
        __syncthreads();
        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpSum[threadIdx.y][threadIdx.x] : 0;
        if (warpIdx == 0)
            sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
        if (threadIdx.x == 0 && unq_idx < num_unq)
            atomicAdd(&d_out[unq_idx * num_dim + dim], sum);
    }
}

bool check(float *calc_cnt, int *gt_cnt, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(calc_cnt[i] - gt_cnt[i]) < 1e-3)
            return false;
    }
    return true;
}

void read_file(std::string filename, std::vector<int> &array, int num_cols) {
    std::ifstream infile(filename.c_str());
    std::string line;
    int word;
    if (!infile) {
        printf("Cannot open test_data.txt");
        exit(1);
    }
    while (std::getline(infile, line)) {
        std::istringstream words(line);
        if (line.length() == 0) {
            continue;
        }
        for (int i = 0; i < num_cols; i++) {
            if (words >> word) {
                array.push_back(word);
            } else {
                printf("Error for reading test_data.txt\n");
                exit(1);
            }
        }
    }
    infile.close();
}

int main() {
    // input
    std::vector<int> unq_preSum, unq_cnt;
    std::string preSum_name = "/home/yuxue_yang/PythonWorks/TorchEx/unq_preSum.txt", cnt_name = "/home/yuxue_yang/PythonWorks/TorchEx/unq_cnt.txt";
    int num_cols = 1;
    read_file(preSum_name, unq_preSum, num_cols);
    read_file(cnt_name, unq_cnt, num_cols);
    int N = unq_cnt.size();
    assert(N + 1 == unq_preSum.size());
    int num_total = 344930, max_cnt = 198;
    float *d_feats;
    int feats_mem = num_total * num_cols * sizeof(float);
    int *d_preSum;
    int preSum_mem = (N + 1) * num_cols * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_preSum, preSum_mem));
    CHECK_CALL(cudaMemcpy(d_preSum, unq_preSum.data(), preSum_mem, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMalloc(&d_feats, feats_mem));
    CHECK_CALL(cudaMemset(d_feats, 1, feats_mem));
    float *d_out;
    int out_mem = N * num_cols * sizeof(float);
    CHECK_CALL(cudaMalloc(&d_out, out_mem));
    GPUTimer calc_timer;
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridSize(DIVUP(max_cnt, THREADS_PER_BLOCK), DIVUP(N, THREADS_PER_BLOCK));
    calc_timer.start();
    scatter_sum<<<gridSize, blockSize>>>(d_feats, d_preSum, d_out, N, num_cols);
    calc_timer.stop();
    float *calc_cnt = new float[N];
    CHECK_CALL(cudaMemcpy(calc_cnt, d_out, out_mem, cudaMemcpyDeviceToHost));
    printf("over, check result is %d\n", check(calc_cnt, unq_cnt.data(), N));
    CHECK_CALL(cudaFree(d_preSum));
    CHECK_CALL(cudaFree(d_feats));
    CHECK_CALL(cudaFree(d_out));
    delete[] calc_cnt;
    return 0;
}