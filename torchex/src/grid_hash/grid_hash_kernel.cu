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

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION


__global__ void build_hash_table_kernel(
    const int *keys,
    const int *values,
    int *table,
    int table_size,
    int N
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int key = keys[idx];
  int value = values[idx];
  setvalue(key, value, table, table_size);

}

__global__ void probe_hash_table_kernel(
    const int *keys,
    const int *table,
    int *out_values,
    int table_size,
    int N
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int key = keys[idx];
  int v = getvalue(key, table, table_size);
  assert (v != -1); // -1 means key error, because all values are >= 0
  out_values[idx] = v;
}

void probe_hash_table_launcher(
  const int *keys,
  const int *table,
  int *out_values,
  int table_size,
  int N
  ) {

  dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  probe_hash_table_kernel<<<blocks, threads>>>(
      keys,
      table,
      out_values,
      table_size,
      N
  );

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}


void build_hash_table_launcher(
  const int *keys,
  const int *values,
  int *table,
  int table_size,
  int N
  ) {

  dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);
  assert (table_size >= N);

  build_hash_table_kernel<<<blocks, threads>>>(
      keys,
      values,
      table,
      table_size,
      N
  );

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}
