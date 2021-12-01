// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <torch/types.h>
#include "../utils/error.cuh"

#define THREADS_PER_BLOCK 256
#define LARGE_NEG -10000
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
#define ASSERTION


__global__ void assign_point2voxel(
  float vs_x,
  float vs_y,
  float vs_z,
  float dilation_x,
  float dilation_y,
  float dilation_z,

  float x_min,
  float y_min,
  float z_min,

  float max_grid_x,
  float max_grid_y,
  float max_grid_z,

  int pts_num,
  int max_pts_each_voxel,
  const float *pts,
  int *pts_idx_per_voxel,
  int *counter,
  ) {
    // pts_idx_per_voxel: [n_voxels, max_points]
    // counter: [n_voxels]
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt_idx >= pts_num) return;
  pts += pts_idx * 3;

  int pts_coor_x = floor((pts[0] - x_min) / vs_x);
  int pts_coor_y = floor((pts[1] - y_min) / vs_x);
  int pts_coor_z = floor((pts[2] - z_min) / vs_x);

  pts_coor_x = std::min(std::max(pts_coor_x, 0), max_grid_x-1);
  pts_coor_y = std::min(std::max(pts_coor_y, 0), max_grid_y-1);
  pts_coor_z = std::min(std::max(pts_coor_z, 0), max_grid_z-1);

  int max_search_x = ceil(dilation_x);
  int max_search_y = ceil(dilation_y);
  int max_search_z = ceil(dilation_z);

  for (int kx = -max_search_x; kx =! max_search_x + 1; kx++){
    for (int ky = -max_search_y; ky =! max_search_y + 1; ky++){
      for (int kz = -max_search_z; kz =! max_search_z + 1; kz++){

        float curr_x_coor = static_cast<float>(std::min(std::max(pts_coor_x + kx, 0), max_grid_x-1));
        float curr_y_coor = static_cast<float>(std::min(std::max(pts_coor_y + ky, 0), max_grid_y-1));
        float curr_z_coor = static_cast<float>(std::min(std::max(pts_coor_z + kz, 0), max_grid_z-1));

        float curr_x = (curr_x_coor + 0.5) * vs_x + x_min;
        if (fabs(curr_x - pts[0]) > 0.5 * vs_x + dilation_x * vs_x) continue;

        float curr_y = (curr_y_coor + 0.5) * vs_y + y_min;
        if (fabs(curr_y - pts[0]) > 0.5 * vs_y + dilation_y * vs_y) continue;

        float curr_z = (curr_z_coor + 0.5) * vs_z + z_min;
        if (fabs(curr_z - pts[0]) > 0.5 * vs_z + dilation_z * vs_z) continue;

        int voxel_idx = curr_x_coor * max_grid_y * max_grid_x + 
                        curr_y_coor * max_grid_z + 
                        curr_z_coor;

        assert(voxel_idx < max_grid_x * max_grid_y * max_grid_z);
        
        int cnt = atomicAdd(&counter[voxel_idx], 1);
        if (cnt >= max_pts_each_voxel) return;

        pts_idx_per_voxel[voxel_idx * max_pts_each_voxel + cnt] = pts_idx;
      }
    }
  }
}


std::vector<at::Tensor> weighted_point2voxel_launcher(
  at::Tensor pts, at::Tensor pts_feature,
  float tau,
  int max_pts_each_voxel,
  std::vector<float> voxel_size,
  std::vector<float> enlarged_voxel_size,
  std::vector<float> pc_range
){
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  int pts_num = pts.size(0);
  float x_min = pc_range[0];
  float y_min = pc_range[1];
  float z_min = pc_range[2];
  float x_max = pc_range[3];
  float y_max = pc_range[4];
  float z_max = pc_range[5];

  float vs_x = voxel_size[0];
  float vs_y = voxel_size[1];
  float vs_z = voxel_size[2];

  float dilation_x = elarged_voxel_size[0];
  float dilation_y = elarged_voxel_size[1];
  float dilation_z = elarged_voxel_size[2];

  int max_grid_x = ceil((x_max - x_min) / vs_x);
  int max_grid_y = ceil((y_max - y_min) / vs_x);
  int max_grid_z = ceil((z_max - z_min) / vs_x);


  int *counter = NULL;
  CHECK_CALL(cudaMalloc(&counter,   max_grid_x * max_grid_y * max_grid_z * sizeof(int)));
  CHECK_CALL(cudaMemset(counter, 0, max_grid_x * max_grid_y * max_grid_z * sizeof(int)));

  int *pts_idx_per_voxel = NULL;
  CHECK_CALL(cudaMalloc(&pts_idx_per_voxel,   max_grid_x * max_grid_y * max_grid_z * max_pts_each_voxel * sizeof(int)));
  CHECK_CALL(cudaMemset(pts_idx_per_voxel, -1, max_grid_x * max_grid_y * max_grid_z * max_pts_each_voxel * sizeof(int)));
  // at::Tensor counter = at::zeros(max_grid_x * max_grid_y * max_grid_z, at::dtype(at::kInt32));
  // at::Tensor pts_idx_per_voxel = -1 * at::ones(max_grid_x * max_grid_y * max_grid_z * max_pts_each_voxel, at::dtype(at::kInt32));

  dim3 blocks_assign(DIVUP(pts_num, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  assign_point2voxel<<<blocks_assign, threads>>>(
      vs_x, vs_y, vs_z, dilation_x, dilation_y, dilation_z,
      x_min, y_min, z_min, max_grid_x, max_grid_y, max_grid_z,
      pts_num, max_pts_each_voxel,
      pts.data<float>(),
      pts_idx_per_voxel,
      counter);


  cudaFree(pts_mask);

#ifdef DEBUG
  cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

__global__ void roiaware_avgpool3d_backward(int boxes_num, int channels,
                                            int out_x, int out_y, int out_z,
                                            int max_pts_each_voxel,
                                            const int *pts_idx_of_voxels,
                                            const float *grad_out,
                                            float *grad_in) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

  int x_idx = voxel_idx_flat / (out_y * out_z);
  int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
  int z_idx = voxel_idx_flat % out_z;
  if (box_idx >= boxes_num || channel_idx >= channels || x_idx >= out_x ||
      y_idx >= out_y || z_idx >= out_z)
    return;

  int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
  pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel +
                       offset_base * max_pts_each_voxel;
  grad_out += box_idx * out_x * out_y * out_z * channels +
              offset_base * channels + channel_idx;

  int total_pts = pts_idx_of_voxels[0];
  float cur_grad = 1 / fmaxf(float(total_pts), 1.0);
  for (int k = 1; k <= total_pts; k++) {
    atomicAdd(grad_in + pts_idx_of_voxels[k] * channels + channel_idx,
              grad_out[0] * cur_grad);
  }
}

void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y,
                                       int out_z, int channels,
                                       int max_pts_each_voxel,
                                       const int *pts_idx_of_voxels,
                                       const int *argmax, const float *grad_out,
                                       float *grad_in, int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool, 1: avg_pool

  dim3 blocks(DIVUP(out_x * out_y * out_z, THREADS_PER_BLOCK), channels,
              boxes_num);
  dim3 threads(THREADS_PER_BLOCK);
  if (pool_method == 0) {
    roiaware_maxpool3d_backward<<<blocks, threads>>>(
        boxes_num, channels, out_x, out_y, out_z, argmax, grad_out, grad_in);
  } else if (pool_method == 1) {
    roiaware_avgpool3d_backward<<<blocks, threads>>>(
        boxes_num, channels, out_x, out_y, out_z, max_pts_each_voxel,
        pts_idx_of_voxels, grad_out, grad_in);
  }
}
