// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/types.h>
#include "cuda_fp16.h"
#include "../utils/error.cuh"

#define THREADS_PER_BLOCK 256
#define LARGE_NEG -10000
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y,
                                             float rz, float &local_x,
                                             float &local_y) {
  // should rotate pi/2 + alpha to translate LiDAR to local
  float rot_angle = rz + M_PI / 2;
  float cosa = cos(rot_angle), sina = sin(rot_angle);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d,
                                        float &local_x, float &local_y,
                                        const float extra_w,
                                        const float extra_l,
                                        const float extra_h
                                        ) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, w, l, h, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float w = box3d[3], l = box3d[4], h = box3d[5], rz = box3d[6];
  float large_w = w + extra_w;
  float large_l = l + extra_l;
  float large_h = h + extra_h;
  cz += h / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > large_h / 2.0) return 0;

  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);

  float in_flag = (local_x > -l / 2.0) & (local_x < l / 2.0) &
                  (local_y > -w / 2.0) & (local_y < w / 2.0) & (fabsf(z - cz) < h / 2.0);

  float in_large_flag = 
                  (local_x > -large_l / 2.0) & (local_x < large_l / 2.0) &
                  (local_y > -large_w / 2.0) & (local_y < large_w / 2.0);
  
  if (in_flag > 0) return 2;
  else if (in_large_flag > 0) return 1;
  else return 0;
  // 0: out of large box
  // 1: in large box but out of small box
  // 2: in small box
}

__global__ void generate_pts_mask_for_box3d(int boxes_num, int pts_num,
                                            int out_x, int out_y, int out_z,
                                            const float *rois, const float *pts,
                                            int *counter,
                                            int pts_feature_dim,
                                            int max_pts_each_voxel,
                                            float extra_w,
                                            float extra_l,
                                            float extra_h,
                                            float *pooled_features
                                          ) {
  // params rois: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z]
  // counter: [N, x, y, z]
  // pooled_features [N, x, y, z, max_points, C]
  // otherwise: encode (x_idxs, y_idxs, z_idxs) by binary bit
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int box_idx = blockIdx.y;
  if (pt_idx >= pts_num || box_idx >= boxes_num) return;

  pts += pt_idx * 3;
  rois += box_idx * 7;

  float local_x = 0, local_y = 0;
  int cur_in_flag = check_pt_in_box3d(pts, rois, local_x, local_y, extra_w, extra_l, extra_h);

  if (cur_in_flag > 0) {

    float w = rois[3], l = rois[4], h = rois[5];
    float local_z = pts[2] - (rois[2] + h / 2.0); // to roi center

    float large_w = w + extra_w;
    float large_l = l + extra_l;
    float large_h = h + extra_h;

    float x_res = large_l / out_x;
    float y_res = large_w / out_y;
    float z_res = large_h / out_z;

    unsigned int x_idx = int((local_x + large_l / 2) / x_res);
    unsigned int y_idx = int((local_y + large_w / 2) / y_res);
    unsigned int z_idx = int((local_z + large_h / 2) / z_res);

    x_idx = min(max(x_idx, 0), out_x - 1);
    y_idx = min(max(y_idx, 0), out_y - 1);
    z_idx = min(max(z_idx, 0), out_z - 1);


    // unsigned int idx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx;
    unsigned int counter_offset = box_idx * out_x * out_y * out_z + 
                                  x_idx * out_y * out_z +
                                  y_idx * out_z +
                                  z_idx;

    int cnt = atomicAdd(&counter[counter_offset], 1);
    if (cnt >= max_pts_each_voxel) return;

    unsigned int feat_offset = counter_offset * max_pts_each_voxel * pts_feature_dim +
                               cnt * pts_feature_dim;


    float voxel_cen_x = (static_cast<float>(x_idx) + 0.5) * x_res - large_l / 2;
    float voxel_cen_y = (static_cast<float>(y_idx) + 0.5) * y_res - large_w / 2;
    float voxel_cen_z = (static_cast<float>(z_idx) + 0.5) * z_res - large_h / 2;

    float rel_x = local_x - voxel_cen_x;
    float rel_y = local_y - voxel_cen_y;
    float rel_z = local_z - voxel_cen_z;

    float off_x = local_x + l / 2;
    float off_y = local_y + w / 2;
    float off_z = local_z + h / 2;

    float off_x_2 = -local_x + l / 2;
    float off_y_2 = -local_y + w / 2;
    float off_z_2 = -local_z + h / 2;

    float is_in_margin = cur_in_flag == 1 ? 1 : 0;

    assert(pts_feature_dim == 16);


    pooled_features[feat_offset + 0 ] = rel_x;
    pooled_features[feat_offset + 1 ] = rel_y;
    pooled_features[feat_offset + 2 ] = rel_z;

    pooled_features[feat_offset + 3 ] = local_x;
    pooled_features[feat_offset + 4 ] = local_y;
    pooled_features[feat_offset + 5 ] = local_z;

    pooled_features[feat_offset + 6 ] = off_x;
    pooled_features[feat_offset + 7 ] = off_y;
    pooled_features[feat_offset + 8 ] = off_z;
    pooled_features[feat_offset + 9 ] = off_x_2;
    pooled_features[feat_offset + 10] = off_y_2;
    pooled_features[feat_offset + 11] = off_z_2;

    pooled_features[feat_offset + 12] = is_in_margin;

    pooled_features[feat_offset + 13] = pts[0]; // save abs coordinates for visualization
    pooled_features[feat_offset + 14] = pts[1];
    pooled_features[feat_offset + 15] = pts[2];

    #ifdef ASSERTION
    if (is_in_margin == 1){
      assert (
        (local_x <= -l/2 + 1e-4) | (local_x >= l/2 - 1e-4) | 
        (local_y <= -w/2 + 1e-4) | (local_y >= w/2 - 1e-4) |
        (local_z <= -h/2 + 1e-4) | (local_z >= h/2 - 1e-4)
      );
      assert (
        ((off_x <= 0 + 1e-4) && (off_x >= -extra_l/2-1e-4)) || ((off_x >= l - 1e-4) && (off_x <= l + extra_l/2 + 1e-4)) ||
        ((off_y <= 0 + 1e-4) && (off_y >= -extra_w/2-1e-4)) || ((off_y >= w - 1e-4) && (off_y <= w + extra_w/2 + 1e-4)) ||
        ((off_z <= 0 + 1e-4) && (off_z >= -extra_h/2-1e-4)) || ((off_z >= h - 1e-4) && (off_z <= h + extra_h/2 + 1e-4))
      );
      assert (
        ((off_x_2 <= 0 + 1e-4) && (off_x_2 >= -extra_l/2-1e-4)) || ((off_x_2 >= l - 1e-4) && (off_x_2 <= l + extra_l/2 + 1e-4)) ||
        ((off_y_2 <= 0 + 1e-4) && (off_y_2 >= -extra_w/2-1e-4)) || ((off_y_2 >= w - 1e-4) && (off_y_2 <= w + extra_w/2 + 1e-4)) ||
        ((off_z_2 <= 0 + 1e-4) && (off_z_2 >= -extra_h/2-1e-4)) || ((off_z_2 >= h - 1e-4) && (off_z_2 <= h + extra_h/2 + 1e-4))
      );
    }
    if (is_in_margin == 2){
      assert(off_x > 0 - 1e-5 && off_x_2 > 0 - 1e-5 && off_y > 0 - 1e-5 && off_y_2 > 0 - 1e-5 && off_z > 0 - 1e-5 && off_z_2 > 0 - 1e-5);
      assert(off_x < l + 1e-5 && off_x_2 < l + 1e-5 && off_y < w + 1e-5 && off_y_2 < w + 1e-5 && off_z < h + 1e-5 && off_z_2 < h + 1e-5);
    }
    #endif

  }
}


void roi_point_voxelization_launcher(int boxes_num, int pts_num, 
                              int max_pts_each_voxel, int out_x, int out_y,
                              int out_z,
                              int pts_feature_dim,
                              const float *rois, const float *pts,
                              int *pts_idx_of_voxels, float *pooled_features, 
                              const float *extra_wlh) {
  // params rois: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, C)
  // params pool_method: 0: max_pool 1: avg_pool

  int *counter = NULL;
  CHECK_CALL(cudaMalloc(&counter,   boxes_num * out_x * out_y * out_z * sizeof(int)));  // (N, M)
  CHECK_CALL(cudaMemset(counter, 0, boxes_num * out_x * out_y * out_z * sizeof(int)));

  dim3 blocks_mask(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
  dim3 threads(THREADS_PER_BLOCK);
  float extra_w = extra_wlh[0];
  float extra_l = extra_wlh[1];
  float extra_h = extra_wlh[2];
  generate_pts_mask_for_box3d<<<blocks_mask, threads>>>(
      boxes_num, pts_num, out_x, out_y, out_z, rois, pts, counter, pts_feature_dim, 
      max_pts_each_voxel, extra_w, extra_l, extra_h, pooled_features);
  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  // TODO: Merge the collect and pool functions, SS

  cudaFree(counter);

}