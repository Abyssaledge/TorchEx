#include "../utils/error.cuh"
#include "../utils/timer.cuh"
#include <assert.h>
#include <fstream>
#include <sstream>
#include <vector>

#define MAX_THREADS 1024
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

template <typename T>
__device__ inline T warpReduceSum(T sum, int blockSize) {
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

__global__ void getPreSum(const int *const unq_inv, int *const preSum, int n) {
    static __shared__ int groupIdx[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = tid + blockIdx.x * blockDim.x;
    groupIdx[tid] = (i < n) ? unq_inv[i] : -1;
    __syncthreads();
    int groupIdx_i = -1, groupIdx_i_ = -1;
    if (i < n - 1) {
        groupIdx_i = groupIdx[tid];
        groupIdx_i_ = (tid == THREADS_PER_BLOCK - 1) ? unq_inv[i + 1] : groupIdx[tid + 1];
    } else if (i == n - 1) {
        groupIdx_i_ = groupIdx_i + 1; // make them unequal
    }
    if (groupIdx_i != groupIdx_i_)
        preSum[groupIdx[tid] + 1] = i + 1;
}

__global__ void getMaxCnt(const int *const preSum, int *d_max_cnt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        atomicMax(d_max_cnt, preSum[i + 1] - preSum[i]);
}

__global__ void scatter_sum(const float *const d_feats, const int *const d_preSum, float *const d_out, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    static __shared__ float warpSum[MAX_THREADS / THREADS_PER_BLOCK][THREADS_PER_BLOCK / WARP_SIZE];
    float sum = 0;
    int begin = -1, end = -1;
    if (unq_idx < num_unq) {
        begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
        for (int dim = 0; dim < num_dim && i == 0; dim++) {
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
        if (threadIdx.x == 0 && unq_idx < num_unq) {
            atomicAdd(&d_out[unq_idx * num_dim + dim], sum);
        }
    }
}

__global__ void scatter_sum_V2(const float *const d_feats, const int *const d_preSum, float *const d_out, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.z * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    static __shared__ float warpSum[MAX_THREADS / THREADS_PER_BLOCK][THREADS_PER_BLOCK / WARP_SIZE];
    float sum = 0;
    int dim = blockIdx.y;
    assert(dim < num_dim);
    int begin = -1, end = -1;
    if (unq_idx < num_unq) {
        begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
        if (i == 0) {
            d_out[unq_idx * num_dim + dim] = 0;
        }
    }
    int feat_idx = begin + i;
    int laneIdx = threadIdx.x % WARP_SIZE;
    int warpIdx = threadIdx.x / WARP_SIZE;
    sum = (feat_idx < end) ? d_feats[feat_idx * num_dim + dim] : 0;
    sum = warpReduceSum(sum, blockDim.x);
    if (laneIdx == 0)
        warpSum[threadIdx.y][warpIdx] = sum;
    __syncthreads();
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpSum[threadIdx.y][threadIdx.x] : 0;
    if (warpIdx == 0)
        sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    if (threadIdx.x == 0 && unq_idx < num_unq) {
        atomicAdd(&d_out[unq_idx * num_dim + dim], sum);
    }
}

void scatter_sum_launcher(const float *const feats, const int *const unq_inv, float *const out, int num_total, int channel, int num_unq) {
    int *d_preSum, max_cnt, *d_max_cnt;
    int preSum_mem = (num_unq + 1) * sizeof(int);
    CHECK_CALL(cudaMalloc(&d_preSum, preSum_mem));
    CHECK_CALL(cudaMalloc(&d_max_cnt, sizeof(int)));
    CHECK_CALL(cudaMemset(d_max_cnt, 0, sizeof(int)));
    getPreSum<<<DIVUP(num_total, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(unq_inv, d_preSum, num_total);
    getMaxCnt<<<DIVUP(num_unq, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_preSum, d_max_cnt, num_unq);
    CHECK_CALL(cudaMemcpy(&max_cnt, d_max_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    dim3 blockSize(THREADS_PER_BLOCK, MAX_THREADS / THREADS_PER_BLOCK);
    dim3 gridSize(DIVUP(max_cnt, blockSize.x), DIVUP(num_unq, blockSize.y));
    scatter_sum<<<gridSize, blockSize>>>(feats, d_preSum, out, num_unq, channel);
    // dim3 blockSize(THREADS_PER_BLOCK, MAX_THREADS / THREADS_PER_BLOCK);
    // dim3 gridSize(DIVUP(max_cnt, blockSize.x), channel, DIVUP(num_unq, blockSize.y));
    // scatter_sum_V2<<<gridSize, blockSize>>>(feats, d_preSum, out, num_unq, channel);
    CHECK_CALL(cudaFree(d_preSum));
    CHECK_CALL(cudaFree(d_max_cnt));
}

// void read_file(std::string filename, std::vector<int> &array, int num_cols) {
//     std::ifstream infile(filename.c_str());
//     std::string line;
//     int word;
//     if (!infile) {
//         printf("Cannot open test_data.txt");
//         exit(1);
//     }
//     while (std::getline(infile, line)) {
//         std::istringstream words(line);
//         if (line.length() == 0) {
//             continue;
//         }
//         for (int i = 0; i < num_cols; i++) {
//             if (words >> word) {
//                 array.push_back(word);
//             } else {
//                 printf("Error for reading test_data.txt\n");
//                 exit(1);
//             }
//         }
//     }
//     infile.close();
// }

// int main() {
//     // input
//     std::vector<int> unq_preSum, unq_inv;
//     std::string preSum_name = "/home/yangyuxue/CudaPractice/connect/unq_preSum_test.txt", inv_name = "/home/yangyuxue/CudaPractice/connect/unq_inv_test.txt";
//     int num_cols = 1;
//     read_file(preSum_name, unq_preSum, num_cols);
//     read_file(inv_name, unq_inv, num_cols);
//     int channel = 128;
//     int num_unq = unq_preSum.size() - 1;
//     int num_total = unq_inv.size();
//     // int *calc_preSum = new int[num_unq + 1];
//     int *d_unq_inv;
//     // int *d_preSum, *d_max_cnt;
//     // int *max_cnt = new int(0);
//     int inv_mem = num_total * num_cols * sizeof(int);
//     // int preSum_mem = (num_unq + 1) * num_cols * sizeof(int);
//     CHECK_CALL(cudaMalloc(&d_unq_inv, inv_mem));
//     CHECK_CALL(cudaMemcpy(d_unq_inv, unq_inv.data(), inv_mem, cudaMemcpyHostToDevice));
//     // CHECK_CALL(cudaMalloc(&d_preSum, preSum_mem));
//     // CHECK_CALL(cudaMalloc(&d_max_cnt, sizeof(int)));
//     // CHECK_CALL(cudaMemset(d_max_cnt, 0, sizeof(int)));
//     float *d_feats, *feats = new float[num_total * channel];
//     for (int i = 0; i < num_total * channel; i++)
//         feats[i] = 1.0;
//     uint feats_mem = num_total * channel * sizeof(float);
//     CHECK_CALL(cudaMalloc(&d_feats, feats_mem));
//     CHECK_CALL(cudaMemcpy(d_feats, feats, feats_mem, cudaMemcpyHostToDevice));
//     float *d_out, *out = new float[num_unq * channel];
//     CHECK_CALL(cudaMalloc(&d_out, num_unq * channel * sizeof(float)));

//     GPUTimer timer;
//     timer.start();
//     scatter_sum_launcher(d_feats, d_unq_inv, d_out, num_total, channel, num_unq);
//     timer.stop();
//     CHECK_CALL(cudaMemcpy(out, d_out, num_unq * channel * sizeof(float), cudaMemcpyDeviceToHost));
//     for (int i = 0; i < num_unq; i++) {
//         for (int j = 0; j < channel; j++) {
//             float delta = unq_preSum[i + 1] - unq_preSum[i];
//             if (abs(out[i * channel] - delta) > 1e-3)
//                 printf("error. out[%3d][%3d]:%3.1f, cnt[i]:%3.1f\n", i, j, out[i * channel + j], delta);
//         }
//     }
//     return 0;
// }