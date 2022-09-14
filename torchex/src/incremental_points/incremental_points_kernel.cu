#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "cuda_fp16.h"
#include "../utils/error.cuh"

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION
__forceinline__ int up_2n(int n){
    if (n == 1) return 1;
    int temp = n - 1;
    temp |= temp >> 1;
    temp |= temp >> 2;
    temp |= temp >> 4;
    temp |= temp >> 8;
    temp |= temp >> 16;
    return temp + 1;
}

__device__ __forceinline__ int double_hash(int key, int probe_i, int table_size){
  // equivalent to (key +  probe_i * (key * 2 + 1)) % table_size, keep the one more mod op for better understanding.
  return (key % table_size +  probe_i * (key * 2 + 1)) % table_size;
}

__global__ void build_hashtable_kernel(
    int *hash_table,
    int table_size,
    const int *base_coors,
    int N_base
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_base) return;
  int key = base_coors[idx];
  for (int i = 0; i < table_size; i++){
    int slot_idx = double_hash(key, i, table_size);
    int old_key = atomicCAS(&hash_table[slot_idx], -1, key);
    if (old_key == -1) break;
    if (old_key == key) break;
  }

}


__global__ void probe_kernel(
    const int *hash_table,
    int  table_size,
    const int *inc_coors,
    bool *out_mask,
    int N_out
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_out) return;
  int key = inc_coors[idx];
  for (int i = 0; i < table_size; i++){
    int slot_idx = double_hash(key, i, table_size);
    int value = hash_table[slot_idx];
    if (value == -1){
      out_mask[idx] = true;
      return;
    }
    else if (value == key) {
      return;
    }
  }
  printf("This should not happen.");
  assert(false);

}


 void incremental_points_launcher(
  const int *base_coors,
  const int *inc_coors,
  bool *out_mask,
  int N_unq,
  int N_base,
  int N_out
  ) {

  const int table_size = up_2n(N_unq) * 2; // make sure hashing load factor < 0.5
  int *hash_table = NULL;
  CHECK_CALL(cudaMalloc(&hash_table,   table_size * sizeof(int)));
  CHECK_CALL(cudaMemset(hash_table, -1, table_size * sizeof(int)));

  dim3 blocks_build_table(DIVUP(N_base, 1024));
  dim3 threads_build_table(1024);

  build_hashtable_kernel<<<blocks_build_table, threads_build_table>>>(
      hash_table,
      table_size,
      base_coors,
      N_base
  );

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif


  dim3 blocks_probe(DIVUP(N_out, THREADS_PER_BLOCK));
  dim3 threads_probe(THREADS_PER_BLOCK);

  probe_kernel<<<blocks_probe, threads_probe>>>(
      hash_table,
      table_size,
      inc_coors,
      out_mask,
      N_out
  );

  cudaFree(hash_table);

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}
