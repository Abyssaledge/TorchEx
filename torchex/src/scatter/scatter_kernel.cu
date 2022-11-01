#include "../utils/error.cuh"
#include "../utils/functions.cuh"
#include "../utils/timer.cuh"
#include <assert.h>
#include <cfloat>
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

template <typename T>
__device__ inline void warpReduceMax(T &max_value, int &idx, int blockSize) {
    if (blockSize >= 32) {
        T temp_max = __shfl_down_sync(0xffffffff, max_value, 16);
        int temp_idx = __shfl_down_sync(0xffffffff, idx, 16);
        if (temp_max > max_value) {
            max_value = temp_max;
            idx = temp_idx;
        }
    }
    if (blockSize >= 16) {
        T temp_max = __shfl_down_sync(0xffffffff, max_value, 8);
        int temp_idx = __shfl_down_sync(0xffffffff, idx, 8);
        if (temp_max > max_value) {
            max_value = temp_max;
            idx = temp_idx;
        }
    }
    if (blockSize >= 8) {
        T temp_max = __shfl_down_sync(0xffffffff, max_value, 4);
        int temp_idx = __shfl_down_sync(0xffffffff, idx, 4);
        if (temp_max > max_value) {
            max_value = temp_max;
            idx = temp_idx;
        }
    }
    if (blockSize >= 4) {
        T temp_max = __shfl_down_sync(0xffffffff, max_value, 2);
        int temp_idx = __shfl_down_sync(0xffffffff, idx, 2);
        if (temp_max > max_value) {
            max_value = temp_max;
            idx = temp_idx;
        }
    }
    if (blockSize >= 2) {
        T temp_max = __shfl_down_sync(0xffffffff, max_value, 1);
        int temp_idx = __shfl_down_sync(0xffffffff, idx, 1);
        if (temp_max > max_value) {
            max_value = temp_max;
            idx = temp_idx;
        }
    }
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void getPreSum(const int *const unq_inv, int *const preSum, int n) {
    extern __shared__ int groupIdx[];
    int tid = threadIdx.x;
    int i = tid + blockIdx.x * blockDim.x;
    groupIdx[tid] = (i < n) ? unq_inv[i] : -1;
    __syncthreads();
    int groupIdx_i = -1, groupIdx_i_ = -1;
    if (i < n - 1) {
        groupIdx_i = groupIdx[tid];
        groupIdx_i_ = (tid == blockDim.x - 1) ? unq_inv[i + 1] : groupIdx[tid + 1];
    } else if (i == n - 1) {
        groupIdx_i_ = groupIdx_i + 1; // make them unequal
    }
    if (groupIdx_i != groupIdx_i_)
        preSum[groupIdx[tid] + 1] = i + 1;
}

__global__ void getUnqCnts32(const int *const unq_cnts, int *const unq_cnts_32, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        unq_cnts_32[i] = DIVUP(unq_cnts[i], 32) * 32;
}

__global__ void getUnqIdx(const int *const last_idx, int *const UnqIdx, int num_unq) {
    int unq_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (unq_idx > 0 && unq_idx < num_unq) {
        int begin = last_idx[unq_idx - 1], end = last_idx[unq_idx];
        for (int i = begin + 1; i <= end; i++) {
            UnqIdx[i] = unq_idx;
        }
    }
}

__global__ void scatter_sum(const float *const d_feats, const int *const d_preSum, float *const d_out, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x;
    int dim = blockIdx.x;
    extern __shared__ float warpMax[];
    int num_valid_warp = DIVUP(blockDim.x, WARP_SIZE);
    float sum = 0;
    int begin = -1, end = -1;
    assert(dim < num_dim);
    if (unq_idx < num_unq) {
        begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
    }
    for (int feat_idx = begin + tid; feat_idx < end; feat_idx += blockDim.x) {
        sum += d_feats[feat_idx * num_dim + dim];
    }
    int laneIdx = tid % WARP_SIZE;
    int warpIdx = tid / WARP_SIZE;
    sum = warpReduceSum(sum, blockDim.x);
    if (laneIdx == 0)
        warpMax[threadIdx.y * num_valid_warp + warpIdx] = sum;
    __syncthreads();
    sum = (tid < num_valid_warp) ? warpMax[threadIdx.y * num_valid_warp + tid] : 0;
    if (warpIdx == 0)
        sum = warpReduceSum(sum, num_valid_warp);
    if (tid == 0 && unq_idx < num_unq) {
        d_out[unq_idx * num_dim + dim] = sum;
    }
}

__global__ void scatter_sumV2(const float *const d_feats, const int *const d_preSum, const int *const d_preSum32, const int *const d_Idx2Unq,
                              float *const d_out, int num_total, int num_total32, int num_unq, int num_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int dim = threadIdx.y + blockIdx.y * blockDim.y;
    float sum = 0;
    int unq = -1;
    if (dim < num_dim) {
        if (i < num_total32) {
            unq = d_Idx2Unq[i];
            int delta = i - d_preSum32[unq];
            int true_idx = delta + d_preSum[unq];
            sum = (true_idx < d_preSum[unq + 1]) ? d_feats[dim * num_total + true_idx] : 0;
        }
        sum = warpReduceSum(sum, blockDim.x);
        int laneIdx = threadIdx.x & (WARP_SIZE - 1);
        if (i < num_total32 && laneIdx == 0)
            atomicAdd(&d_out[dim * num_unq + unq], sum);
    }
}

__global__ void scatter_sumV3(const float *const d_feats, const int *const d_preSum_extend, const int *const d_UnqIdx, float *const d_out, int num_unq_extend, int num_dim) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    int Idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (dim < num_dim && Idx_y < num_unq_extend) {
        int unq_idx = d_UnqIdx[Idx_y];
        int begin = d_preSum_extend[Idx_y], end = d_preSum_extend[Idx_y + 1];
        float sum = 0;
        for (int feat_idx = begin; feat_idx < end; feat_idx++) {
            sum += d_feats[feat_idx * num_dim + dim];
        }
        atomicAdd(&d_out[unq_idx * num_dim + dim], sum);
    }
}

__global__ void scatter_max(const float *const d_feats, const int *const d_preSum, float *const d_out, int *const d_arg, int num_unq, int num_dim) {
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x;
    int dim = blockIdx.x;
    int num_valid_warp = DIVUP(blockDim.x, WARP_SIZE);
    extern __shared__ float shared_mem[];
    float *warpMax = shared_mem;
    int *warpMaxIdx = (int *)&warpMax[blockDim.y * num_valid_warp];
    float max_value = -FLT_MAX;
    int max_idx = -1;
    int begin = -1, end = -1;
    assert(dim < num_dim);
    if (unq_idx < num_unq) {
        begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
    }
    for (int feat_idx = begin + tid; feat_idx < end; feat_idx += blockDim.x) {
        float temp_feat = d_feats[feat_idx * num_dim + dim];
        if (temp_feat >= max_value) {
            max_value = temp_feat;
            max_idx = feat_idx;
        }
    }
    int laneIdx = tid % WARP_SIZE;
    int warpIdx = tid / WARP_SIZE;
    warpReduceMax(max_value, max_idx, blockDim.x);
    if (laneIdx == 0) {
        warpMax[threadIdx.y * num_valid_warp + warpIdx] = max_value;
        warpMaxIdx[threadIdx.y * num_valid_warp + warpIdx] = max_idx;
    }
    __syncthreads();
    if (tid < num_valid_warp) {
        max_value = warpMax[threadIdx.y * num_valid_warp + tid];
        max_idx = warpMaxIdx[threadIdx.y * num_valid_warp + tid];
    }
    if (warpIdx == 0)
        warpReduceMax(max_value, max_idx, num_valid_warp);
    if (tid == 0 && unq_idx < num_unq) {
        d_out[unq_idx * num_dim + dim] = max_value;
        d_arg[unq_idx * num_dim + dim] = max_idx;
    }
}

__global__ void scatter_maxV3_infer(const float *const d_feats, const int *const d_preSum_extend, const int *const d_UnqIdx, float *const d_out,
                              float *const d_temp_max, int *const d_temp_arg, int num_unq_extend, int num_dim) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    int Idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (dim < num_dim && Idx_y < num_unq_extend) {
        int unq_idx = d_UnqIdx[Idx_y];
        int begin = d_preSum_extend[Idx_y], end = d_preSum_extend[Idx_y + 1];
        float max_value = -FLT_MAX;
        float temp_feat;
        int max_idx = -1;
        for (int feat_idx = begin; feat_idx < end; feat_idx++) {
            temp_feat = d_feats[feat_idx * num_dim + dim];
            if (temp_feat > max_value) {
                max_value = temp_feat;
                max_idx = feat_idx;
            }
        }
        d_temp_max[Idx_y * num_dim + dim] = max_value;
        d_temp_arg[Idx_y * num_dim + dim] = max_idx;
        atomicMaxFloat(&d_out[unq_idx * num_dim + dim], max_value);
    }
}

__global__ void scatter_argmaxV3_infer(const int *const d_UnqIdx, const float *const d_temp_max, const int *const d_temp_arg, const float *const d_out, int *const d_arg,
                                 int num_unq_extend, int num_dim) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    int Idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (dim < num_dim && Idx_y < num_unq_extend) {
        int unq_idx = d_UnqIdx[Idx_y];
        if (fabs(d_temp_max[Idx_y * num_dim + dim] - d_out[unq_idx * num_dim + dim]) < FLT_EPSILON)
            d_arg[unq_idx * num_dim + dim] = d_temp_arg[Idx_y * num_dim + dim];
    }
}

__global__ void scatter_maxV3_train(const float *const d_feats, const int *const d_preSum, float *const d_out, int *const d_arg, int num_unq, int num_dim) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    int unq_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (dim < num_dim && unq_idx < num_unq) {
        int begin = d_preSum[unq_idx], end = d_preSum[unq_idx + 1];
        float max_value = -FLT_MAX;
        float temp_feat;
        int max_idx = -1;
        for (int feat_idx = begin; feat_idx < end; feat_idx++) {
            temp_feat = d_feats[feat_idx * num_dim + dim];
            if (temp_feat > max_value) {
                max_value = temp_feat;
                max_idx = feat_idx;
            }
        }
        d_out[unq_idx * num_dim + dim] = max_value;
        d_arg[unq_idx * num_dim + dim] = max_idx;
    }
}

void getPreSum_launcher(const int *const unq_inv, int *const preSum, int num_total) {
    getPreSum<<<DIVUP(num_total, THREADS_PER_BLOCK), THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(unq_inv, preSum, num_total);
}

void getUnqCnts32_launcher(const int *const unq_cnts, int *const unq_cnts32, int num_unq) {
    getUnqCnts32<<<DIVUP(num_unq, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(unq_cnts, unq_cnts32, num_unq);
}

void getUnqIdx_launcher(const int *const last_idx, int *const UnqIdx, int num_unq) {
    getUnqIdx<<<DIVUP(num_unq, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(last_idx, UnqIdx, num_unq);
}

void scatter_sum_launcher(const float *const feats, const int *const preSum, float *const out,
                          int channel, int num_unq, int max_cnt) {
    int max_2n = max(min(up_2n(max_cnt), MAX_THREADS), 32);
    dim3 blockSize(max_2n, MAX_THREADS / max_2n);
    dim3 gridSize(channel, DIVUP(num_unq, blockSize.y));
    scatter_sum<<<gridSize, blockSize, blockSize.y * DIVUP(max_2n, WARP_SIZE) * sizeof(float)>>>(feats, preSum, out, num_unq, channel);
}

void scatter_sumV2_launcher(const float *const feats, const int *const preSum, const int *const preSum32, const int *const Idx2Unq,
                            float *const out, int num_total, int num_total32, int num_unq, int channel, int blockDim_x) {
    dim3 blockSize(blockDim_x, DIVUP(MAX_THREADS, blockDim_x));
    dim3 gridSize(DIVUP(num_total32, blockDim_x), DIVUP(channel, blockSize.y));
    scatter_sumV2<<<gridSize, blockSize>>>(feats, preSum, preSum32, Idx2Unq, out, num_total, num_total32, num_unq, channel);
}

void scatter_sumV3_launcher(const float *const feats, const int *const preSum_extend, const int *const UnqIdx, float *const out,
                            int channel, int num_unq_extend) {
    int num_dim = min(DIVUP(channel, 32) * 32, 128);
    dim3 blockSize(num_dim, MAX_THREADS / num_dim);
    dim3 gridSize(DIVUP(channel, blockSize.x), DIVUP(num_unq_extend, blockSize.y));
    scatter_sumV3<<<gridSize, blockSize>>>(feats, preSum_extend, UnqIdx, out, num_unq_extend, channel);
}

void scatter_max_launcher(const float *const feats, const int *const preSum, float *const out, int *const arg,
                          int channel, int num_unq, int max_cnt) {
    int max_2n = max(min(up_2n(max_cnt), MAX_THREADS), 32);
    dim3 blockSize(max_2n, MAX_THREADS / max_2n);
    dim3 gridSize(channel, DIVUP(num_unq, blockSize.y));
    int shared_mem = blockSize.y * DIVUP(max_2n, WARP_SIZE) * (sizeof(float) + sizeof(int));
    scatter_max<<<gridSize, blockSize, shared_mem>>>(feats, preSum, out, arg, num_unq, channel);
}

void scatter_maxV3_infer_launcher(const float *const feats, const int *const preSum_extend, const int *const UnqIdx, float *const out, int *const arg,
                            int channel, int num_unq_extend) {
    float *temp_max;
    int *temp_argmax;
    int max_mem = num_unq_extend * channel * sizeof(float), argmax_mem = num_unq_extend * channel * sizeof(int);
    CHECK_CALL(cudaMalloc(&temp_max, max_mem));
    CHECK_CALL(cudaMalloc(&temp_argmax, argmax_mem));
    int num_dim = min(DIVUP(channel, 32) * 32, 128);
    dim3 blockSize(num_dim, MAX_THREADS / num_dim);
    dim3 gridSize(DIVUP(channel, blockSize.x), DIVUP(num_unq_extend, blockSize.y));
    scatter_maxV3_infer<<<gridSize, blockSize>>>(feats, preSum_extend, UnqIdx, out, temp_max, temp_argmax, num_unq_extend, channel);
    scatter_argmaxV3_infer<<<gridSize, blockSize>>>(UnqIdx, temp_max, temp_argmax, out, arg, num_unq_extend, channel);
    CHECK_CALL(cudaFree(temp_max));
    CHECK_CALL(cudaFree(temp_argmax));
}

void scatter_maxV3_train_launcher(const float *const feats, const int *const preSum, float *const out, int *const arg,
                          int channel, int num_unq) {
    dim3 blockSize(128, MAX_THREADS / 128);
    dim3 gridSize(DIVUP(channel, blockSize.x), DIVUP(num_unq, blockSize.y));
    scatter_maxV3_train<<<gridSize, blockSize>>>(feats, preSum, out, arg, num_unq, channel);
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
//         float delta = unq_preSum[i + 1] - unq_preSum[i];
//         for (int j = 0; j < channel; j++) {
//             if (abs(out[i * channel] - delta) > 1e-3)
//                 printf("error. out[%3d][%3d]:%3.1f, cnt[i]:%3.1f\n", i, j, out[i * channel + j], delta);
//         }
//     }
//     return 0;
// }