#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "cuda_fp16.h"
#include "../utils/error.cuh"
#include "../utils/functions.cuh"

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION

// I don't know if it is correct to use float &m
__device__ __forceinline__ void warpReduceMax(float m, int ind, float &ret_m, int &ret_ind, unsigned int reduce_range, unsigned int w){
    float tmp_m = 0;
    int tmp_ind = -1;
    bool bigger = false;
    if(reduce_range >= 32){
      tmp_m = __shfl_down_sync(0xffffffff, m, 16, w);
      tmp_ind = __shfl_down_sync(0xffffffff, ind, 16, w);
      bigger = m > tmp_m;
      ind = bigger ? ind: tmp_ind;
      m = bigger ? m: tmp_m;
    } 
    if(reduce_range >= 16){
      tmp_m = __shfl_down_sync(0xffffffff, m, 8, w);
      tmp_ind = __shfl_down_sync(0xffffffff, ind, 8, w);
      bigger = m > tmp_m;
      ind = bigger ? ind: tmp_ind;
      m = bigger ? m: tmp_m;
    } 
    if(reduce_range >= 8){
      tmp_m = __shfl_down_sync(0xffffffff, m, 4, w);
      tmp_ind = __shfl_down_sync(0xffffffff, ind, 4, w);
      bigger = m > tmp_m;
      ind = bigger ? ind: tmp_ind;
      m = bigger ? m: tmp_m;
    } 
    if(reduce_range >= 4){
      tmp_m = __shfl_down_sync(0xffffffff, m, 2, w);
      tmp_ind = __shfl_down_sync(0xffffffff, ind, 2, w);
      bigger = m > tmp_m;
      ind = bigger ? ind: tmp_ind;
      m = bigger ? m: tmp_m;
    } 
    if(reduce_range >= 2){
      tmp_m = __shfl_down_sync(0xffffffff, m, 1, w);
      tmp_ind = __shfl_down_sync(0xffffffff, ind, 1, w);
      bigger = m > tmp_m;
      ind = bigger ? ind: tmp_ind;
      m = bigger ? m: tmp_m;
    } 
    ret_m = m;
    ret_ind = ind;
}


__global__ void assign_block_kernel(
    const int *group_inds,
    int *group_beg_inds,
    int *group_end_inds,
    int *group_cnt,
    int *unq_group_inds,
    int N,
    int table_size
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  if (idx == 0){
    setvalue(group_inds[0], 0, group_beg_inds, table_size);
    setvalue(group_inds[N-1], N, group_end_inds, table_size);
    // group_cnt[0] = 1;
    unq_group_inds[0] = group_inds[0];
  }

  // __syncthreads();

  if (idx > 0){
    int this_group_id = group_inds[idx];
    int pre_group_id = group_inds[idx - 1];
    if (this_group_id != pre_group_id){
      int cnt = atomicAdd(&group_cnt[0], 1) + 1; // a little tricky, because initialized value of group_cnt is 0, but actually
                                                 // it should be set to 1 when idx == 0. 
      // group_beg_inds[cnt] = idx;
      setvalue(this_group_id, idx, group_beg_inds, table_size);
      setvalue(pre_group_id, idx, group_end_inds, table_size);
      unq_group_inds[cnt] = this_group_id;
    }

  }
}

__global__ void group_fps_kernel(
    const float *points, // N
    const int *group_inds,
    const int *group_beg_inds, 
    const int *group_end_inds, 
    const int *unq_group_inds,
    bool *out_mask, // N
    float *dist_to_selected, // N
    int N,
    int K,
    int num_groups,
    int table_size
) {

  int block_size = blockDim.x;
  int tid = threadIdx.x;
  // int pts_idx = blockIdx.x * blockDim.x + tid;

  __shared__ int group_beg_pos, group_end_pos, group_size, num_iters;
  if (tid == 0){
    int this_group_id = unq_group_inds[blockIdx.x];
    assert (this_group_id >= 0);
    group_beg_pos = getvalue(this_group_id, group_beg_inds, table_size);
    group_end_pos = getvalue(this_group_id, group_end_inds, table_size);;

    group_size = group_end_pos - group_beg_pos;
    num_iters = DIVUP(group_size, block_size);
  }
  __syncthreads();
  // if (group_idx == num_groups - 1) group_end_pos = N;
  // else if (group_idx < num_groups - 1) group_end_pos = group_beg_inds[group_idx + 1];
  // else assert(false);


  assert (group_size > 0);
  assert (group_beg_pos >= 0 && group_beg_pos < N);
  assert (group_end_pos > 0 && group_end_pos <= N);

  // do not apply FPS if K >= group_size
  if (K >= group_size){
    for (int iter = 0; iter < num_iters; iter++){
      int offset = tid + iter * block_size;
      if (offset < group_size) out_mask[group_beg_pos + offset] = true;
    }
    return;
  }

  __shared__ int selected;
  __shared__ int counter;

  __shared__ int warpsize_farthest_inds[THREADS_PER_BLOCK / 32];
  __shared__ float warpsize_farthest_dist[THREADS_PER_BLOCK / 32];
  int num_warps = THREADS_PER_BLOCK / 32;

  assert(THREADS_PER_BLOCK == block_size && THREADS_PER_BLOCK <= 1024); // warp reduce requires num_threads < 1024
  if (tid == 0) {
    selected = 0;
    counter = 0;
  }
  // if (tid < N) dist_to_selected[tid] = 1e8;
  for (int iter = 0; iter < num_iters; iter++){
      int group_offset = tid + iter * block_size;
      int global_offset = group_beg_pos + group_offset;
      if (group_offset < group_size){
        dist_to_selected[global_offset] = 1e8;
      }
  }

  __syncthreads();
  int laneID = tid % 32;
  int warpID = tid / 32;
  int ret_ind = -1;
  float ret_dist = 0;

  for (int k_i = 0; k_i < K; k_i++){

    int farthest_inds = -1;
    float farthest_dist = 0;
    int select_pts_offset = group_beg_pos + selected;
    float x1 = points[select_pts_offset * 3 + 0];
    float y1 = points[select_pts_offset * 3 + 1];
    float z1 = points[select_pts_offset * 3 + 2];

    for (int iter = 0; iter < num_iters; iter++){
        int group_offset = tid + iter * block_size;
        int global_offset = group_beg_pos + group_offset;

        // Not necessary to restrict (x2, y2, z2) to the set of unselected points.
        // Because the distance from visited points to the visited set is always zeros theoritically,
        // so the visited points well not be repeatedly selected.
        if (group_offset < group_size){

          float x2 = points[global_offset * 3 + 0];
          float y2 = points[global_offset * 3 + 1];
          float z2 = points[global_offset * 3 + 2];
          float dist_to_new_selection = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
          float dist = min(dist_to_new_selection, dist_to_selected[global_offset]);
          dist_to_selected[global_offset] = dist;
          farthest_inds = dist > farthest_dist ? group_offset : farthest_inds;
          farthest_dist = dist > farthest_dist ? dist : farthest_dist;
        }
    }
    
    // reduce to get currently farthest point to the visited set.
    warpReduceMax(farthest_dist, farthest_inds, ret_dist, ret_ind, 32, 32);
    if (laneID == 0) {
      warpsize_farthest_dist[warpID] = ret_dist;
      warpsize_farthest_inds[warpID] = ret_ind;
    }
    __syncthreads();

    if (warpID == 0){
      if (laneID < num_warps){
        farthest_dist = warpsize_farthest_dist[laneID];
        farthest_inds = warpsize_farthest_inds[laneID];
      }
      else {
        farthest_dist = 0;
        farthest_inds = -1;
      }
      warpReduceMax(farthest_dist, farthest_inds, ret_dist, ret_ind, 32, 32);
      if (laneID == 0) {
        out_mask[group_beg_pos + ret_ind] = true;
        selected = ret_ind;
        counter += 1;
      }
    }
    __syncthreads();
  
  }

}


 void group_fps_launcher(
  const float *points,
  const int *group_inds,
  bool *out_mask,
  int N,
  int K,
  int num_groups_gt
  ) {

  int *group_counter = NULL;
  CHECK_CALL(cudaMalloc(&group_counter, sizeof(int)));
  CHECK_CALL(cudaMemset(group_counter, 0, sizeof(int)));

  int table_size = up_2n(N);

  int *group_beg_inds = NULL;
  CHECK_CALL(cudaMalloc(&group_beg_inds, table_size * 2 * sizeof(int))); // real length is num_groups, allocate max size here
  CHECK_CALL(cudaMemset(group_beg_inds, -1, table_size * 2 * sizeof(int)));

  int *group_end_inds = NULL;
  CHECK_CALL(cudaMalloc(&group_end_inds, table_size * 2 * sizeof(int))); // real length is num_groups, allocate max size here
  CHECK_CALL(cudaMemset(group_end_inds, -1, table_size * 2 * sizeof(int)));

  int *unq_group_inds = NULL;
  CHECK_CALL(cudaMalloc(&unq_group_inds, N * sizeof(int))); // real length is num_groups, allocate max size here
  CHECK_CALL(cudaMemset(unq_group_inds, -1, N * sizeof(int)));


  dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  assign_block_kernel<<<blocks, threads>>>(
      group_inds,
      group_beg_inds,
      group_end_inds,
      group_counter,
      unq_group_inds,
      N,
      table_size
  );

  int num_groups[1];
  CHECK_CALL(cudaMemcpy(num_groups, group_counter, 1, cudaMemcpyDeviceToHost));
  num_groups[0] += 1;
  // assert (num_groups_gt == num_groups[0]);


  dim3 blocks_fps(num_groups[0]);
  dim3 threads_fps(THREADS_PER_BLOCK);

  float *dist_to_selected = NULL;
  CHECK_CALL(cudaMalloc(&dist_to_selected, N * sizeof(float)));
  // CHECK_CALL(cudaMemset(dist_to_selected, 7, N * sizeof(float)));

  group_fps_kernel<<<blocks_fps, threads_fps>>>(
    points,
    group_inds,
    group_beg_inds,
    group_end_inds,
    unq_group_inds,
    out_mask,
    dist_to_selected, N, K, num_groups[0], table_size);

  cudaFree(group_counter);
  cudaFree(group_beg_inds);
  cudaFree(group_end_inds);
  cudaFree(unq_group_inds);
  cudaFree(dist_to_selected);

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}
