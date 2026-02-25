/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/



// headers in STL
#include <stdio.h>

// headers in local files
#include "common.h"
#include "preprocess.h"

__global__ void make_pillar_histo_kernel(
    const float* dev_points, float* dev_pillar_point_feature_in_coors,
    int* pillar_count_histo, const int num_points,
    const int max_points_per_pillar, const int grid_x_size,
    const int grid_y_size, const int grid_z_size, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size,
    const int num_point_feature) {
  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    int count =
        atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1);
    if (count < max_points_per_pillar) {
      int ind =
          y_coor * grid_x_size * max_points_per_pillar * num_point_feature +
          x_coor * max_points_per_pillar * num_point_feature +
          count * num_point_feature;
      // TODO: 只支持num_point_feature=4的实现
      dev_pillar_point_feature_in_coors[ind] = dev_points[th_i * num_point_feature];
      dev_pillar_point_feature_in_coors[ind + 1] = dev_points[th_i * num_point_feature + 1];
      dev_pillar_point_feature_in_coors[ind + 2] = dev_points[th_i * num_point_feature + 2];
      dev_pillar_point_feature_in_coors[ind + 3] = dev_points[th_i * num_point_feature + 3];
    }
  }
}

__global__ void make_pillar_index_kernel(
    int* dev_pillar_count_histo, int* dev_counter,  int* dev_x_coors,
    int* dev_y_coors, float* dev_num_points_per_pillar,
    int* dev_sparse_pillar_map, const int max_pillars,
    const int max_points_per_pillar, const int grid_x_size,
    const int num_inds_for_scan) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomicAdd(dev_counter, 1);
  if (count < max_pillars) {
    // atomicAdd(dev_pillar_count, 1);
    dev_num_points_per_pillar[count] =
      min(num_points_at_this_pillar, max_points_per_pillar);
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;
    // dev_sparse_pillar_map[y * num_inds_for_scan + x] = 1;
    // #ifndef NDEBUG
    // if (count == 0){
    // unsigned long long start = clock64();
    // while (clock64() - start < 50000000ULL) {
    //     // busy wait
    // }
    //   printf("非空pillar个数%d, 已处理pillar个数%d\n", *dev_counter, *dev_pillar_count);
    // }
    //  #endif
    
  }
}

__global__ void make_pillar_feature_kernel(
    float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
    float* dev_pillar_coors, int* dev_x_coors, int* dev_y_coors,
    float* dev_num_points_per_pillar, const int max_points,
    const int num_point_feature, const int grid_x_size) {
  int ith_pillar = blockIdx.x;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = threadIdx.x;
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  int pillar_ind = ith_pillar * max_points * num_point_feature +
                   ith_point * num_point_feature;
  int coors_ind = y_ind * grid_x_size * max_points * num_point_feature +
                  x_ind * max_points * num_point_feature +
                  ith_point * num_point_feature;


  // #pragma unroll 
  // for (int i = 0; i < num_point_feature; ++i) {
  //   dev_pillar_point_feature[pillar_ind + i] = // dev_pillar_point_feature输出不固定
  //       dev_pillar_point_feature_in_coors[coors_ind + i]; 
  // }

  // UGLY num_point_feature需要为编译期常量来支持自动展开，目前手动展开num_point_feature=4
  dev_pillar_point_feature[pillar_ind] = // dev_pillar_point_feature输出不固定
  dev_pillar_point_feature_in_coors[coors_ind]; 
  dev_pillar_point_feature[pillar_ind + 1] = 
  dev_pillar_point_feature_in_coors[coors_ind + 1]; 
  dev_pillar_point_feature[pillar_ind + 2] = 
  dev_pillar_point_feature_in_coors[coors_ind + 2]; 
  dev_pillar_point_feature[pillar_ind + 3] =
  dev_pillar_point_feature_in_coors[coors_ind + 3]; 

  float coor_x = static_cast<float>(x_ind);
  float coor_y = static_cast<float>(y_ind);
  if (ith_point == 0) {
    dev_pillar_coors[ith_pillar * 4 + 0] = 0;
    dev_pillar_coors[ith_pillar * 4 + 1] = 0;
    dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
    dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
  }
  // #ifndef NDEBUG
  // __syncthreads();
  // if (ith_pillar == 0 and ith_point==0){
  //   printf("voxels 中第0个 pillar的坐标为 x=%d,y=%d\n", x_ind, y_ind);
  // }
  // #endif
  }

__global__ void pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_per_pillar) {

    extern __shared__ float temp[];
    int ith_pillar = blockIdx.x; 
    int ith_point  = threadIdx.x;
    int axis = threadIdx.y;
  
    int reduce_size = max_points_per_pillar > 32 ? 64 : 32;
    temp[threadIdx.x * 3 + axis] =  dev_pillar_point_feature[ith_pillar * max_points_per_pillar * num_point_feature + ith_point * num_point_feature + axis];  
    if (threadIdx.x < reduce_size - max_points_per_pillar) {
        temp[(threadIdx.x + max_points_per_pillar) * 3 + axis] = 0.0f; //--> dummy placeholds will set as 0
    }
    __syncthreads();
    int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
    
    /* LEARN: https://stackoverflow.com/questions/6666382/can-i-use-syncthreads-after-having-dropped-threads
    不要在包含 __syncthreads() 的后续代码之前对部分线程 return。 这是 CUDA 的硬性要求。
    但是 In short: yes it's safe.
    The accepted answer may well have been correct when written, but at least since Volta, it is wrong. 
    CUDA docs make clear that the __syncthreads call must be reached by all non-exited threads, 
    which means one can exit early and not cause deadlock. */

    if (ith_point >= num_points_at_this_pillar) {
          return;
    }

    // TODO: Reduction 写法优化
    for (unsigned int d = reduce_size >> 1 ; d > 0; d >>= 1) {
        if (ith_point < d) {
            temp[ith_point*3 +axis] += temp[(ith_point + d) * 3 + axis];
        }
        __syncthreads();
    }

    if (ith_point == 0) {
        dev_points_mean[ith_pillar * 3 + axis] = temp[ith_point + axis] / num_points_at_this_pillar ;
    }
}

__device__ void warpReduce(volatile float* sdata , int ith_point , int axis) {
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 8) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 4) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 2) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 1) * blockDim.y + axis];
}

__global__ void make_pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_pre_pillar) {
    extern __shared__ float temp[];
    unsigned int ith_pillar = blockIdx.x;  // { 0 , 1, 2, ... , 10000+}
    unsigned int ith_point  = threadIdx.x; // { 0 , 1, 2, ...,9}
    unsigned int axis = threadIdx.y; 
    unsigned int idx_pre  = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + ith_point  * num_point_feature;
    unsigned int idx_post = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + (ith_point + blockDim.x)  * num_point_feature;

    temp[ith_point * blockDim.y + axis] = 0.0;
    unsigned int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

    // if (ith_point < num_points_at_this_pillar / 2) {
      temp[ith_point * blockDim.y + axis] = dev_pillar_point_feature[idx_pre  + axis] 
                                          + dev_pillar_point_feature[idx_post + axis];
    // }
    __syncthreads();

    // do reduction in shared mem
    // Sequential addressing. This solves the bank conflicts as
    // the threads now access shared memory with a stride of one
    // 32-bit word (unsigned int) now, which does not cause bank 
    // conflicts
    warpReduce(temp , ith_point , axis);

	// // write result for this block to global mem
    if (ith_point == 0)
    dev_points_mean[ith_pillar * blockDim.y + axis] = temp[ith_point * blockDim.y + axis] / num_points_at_this_pillar ;
}

__global__ void gather_point_feature_kernel(
  const int max_num_pillars_,const int max_num_points_per_pillar,const int num_point_feature,
  const int num_gather_feature, const float min_x_range, const float min_y_range, const float min_z_range, 
  const float pillar_x_size,  const float pillar_y_size, const float pillar_z_size,
  const float* dev_pillar_point_feature, const float* dev_num_points_per_pillar, 
  const float* dev_pillar_coors,
  float* dev_points_mean, 
  float* dev_pfe_gather_feature_){

  int ith_pillar = blockIdx.x; 
  int ith_point = threadIdx.x;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

  if (ith_point >= num_points_at_this_pillar){
        return;
    }

    // UGLY num_gather_feature需要为编译期常量来支持自动展开，目前手动展开num_gather_feature=10
    int base_in  = ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature;
    int base_out = ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature;

    dev_pfe_gather_feature_[base_out] = dev_pillar_point_feature[base_in]; 
  
    dev_pfe_gather_feature_[base_out + 1] = dev_pillar_point_feature[base_in + 1];
  
    dev_pfe_gather_feature_[base_out + 2] = dev_pillar_point_feature[base_in + 2];
  
    dev_pfe_gather_feature_[base_out + 3] = dev_pillar_point_feature[base_in + 3];

    // dev_pfe_gather_feature_[base_out + 4]  =  0.0f;
    //   f_cluster = voxel_features[:, :, :3] - points_mean
    dev_pfe_gather_feature_[base_out + 4] = dev_pillar_point_feature[base_in] - dev_points_mean[ith_pillar * 3];

    dev_pfe_gather_feature_[base_out + 5] = dev_pillar_point_feature[base_in + 1] - dev_points_mean[ith_pillar * 3 + 1];
  
    dev_pfe_gather_feature_[base_out + 6] = dev_pillar_point_feature[base_in + 2] - dev_points_mean[ith_pillar * 3 + 2];

    // f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
    dev_pfe_gather_feature_[base_out + 7]  
    =  dev_pillar_point_feature[base_in] - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + (pillar_x_size/2 + min_x_range));
  
    dev_pfe_gather_feature_[base_out + 8]  
    =  dev_pillar_point_feature[base_in + 1] - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + (pillar_y_size/2 + min_y_range));
  
    dev_pfe_gather_feature_[base_out + 9] 
    =  dev_pillar_point_feature[base_in + 2] - (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size + (pillar_z_size/2 + min_z_range));
    // #ifndef NDEBUG
    // __syncthreads();
    // if (ith_pillar == 0 and ith_point == 0){
    //   for (int i = 0; i < num_gather_feature; ++i){
    //     printf("第 %d 个 pillar 中 point0 第 %i 个元素的值为%.8f \n", ith_pillar, i, dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + i] );
    //   }
    // }
    // #endif
  
}

__global__ void dynamic_voxelize_step1_kernel(
  const float* dev_points, int* coors, const float pillar_x_size, const float pillar_y_size,
  const float pillar_z_size, const float min_x_range, const float min_y_range,
  const float min_z_range, const int grid_x_size, const int grid_y_size, const int grid_z_size,
  const int num_points, const int num_point_feature, int* voxel_num, int* svid_to_dvid_map,
  const int num_gather_feature, float* dev_pfe_gather_feature_, int* voxel_coors) {

    int p_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_id >= num_points) {
      return;
    }
    // To save some computation
    const float* points_offset = dev_points + p_id * num_point_feature;
    int* coors_offset = coors + p_id * 3;
    
    // 计算体素坐标
    int c_x = floorf((points_offset[0] - min_x_range) / pillar_x_size);
    int c_y = floorf((points_offset[1] - min_y_range) / pillar_y_size);
    int c_z = floorf((points_offset[2] - min_z_range) / pillar_z_size);

    float* feat_offset = dev_pfe_gather_feature_ + p_id * num_gather_feature;
    feat_offset[0] = points_offset[0];
    feat_offset[1] = points_offset[1];
    feat_offset[2] = points_offset[2];
    feat_offset[3] = 0; // timestamp为0
    feat_offset[4] = points_offset[0];
    feat_offset[5] = points_offset[1];
    feat_offset[6] = points_offset[2];
    feat_offset[7] = points_offset[0] - (c_x * pillar_x_size + (pillar_x_size * 0.5f + min_x_range));
    feat_offset[8] = points_offset[1] - (c_y * pillar_y_size + (pillar_y_size * 0.5f + min_y_range));
    feat_offset[9] = points_offset[2] - (c_z * pillar_z_size + (pillar_z_size * 0.5f + min_z_range));

    // 先检查坐标有效性，避免无效点写入特征
    if (c_x < 0 || c_x >= grid_x_size || c_y < 0 || c_y >= grid_y_size || c_z < 0 || c_z >= grid_z_size) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
      return;
    }
    
    // 坐标有效，写入coors
    coors_offset[0] = c_z;
    coors_offset[1] = c_y;
    coors_offset[2] = c_x;

    // 计算稀疏体素ID 
    int sparse_voxel_id = (c_z * grid_y_size + c_y) * grid_x_size + c_x; 
    int voxel_flag = atomicCAS(&svid_to_dvid_map[sparse_voxel_id], -1, -2); 
    if (voxel_flag == -1) { 
      int dense_voxel_id = atomicAdd(voxel_num, 1); 
      svid_to_dvid_map[sparse_voxel_id] = dense_voxel_id;
      // 记录稠密体素在 BEV 平面上的 (y, x) 坐标，供后续 scatter 使用
      voxel_coors[dense_voxel_id * 2 + 0] = c_y;
      voxel_coors[dense_voxel_id * 2 + 1] = c_x;
    } 
}

__global__ void dynamic_voxelize_step2_kernel(
  const float* dev_points, int* coors, float* voxel_feats, int* voxel_count_list, 
  const int num_points, const int num_point_feature, int* pid_to_dvid_map, int* svid_to_dvid_map,
  const int grid_x_size, const int grid_y_size) {

    int p_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_id >= num_points) {
      return;
    }
    // To save some computation
    int* coors_offset = coors + p_id * 3;

    int c_z = coors_offset[0];
    if (c_z == -1){
      pid_to_dvid_map[p_id] = -1;
      return;
    }
    int c_y = coors_offset[1];
    int c_x = coors_offset[2];

    int sparse_voxel_id = (c_z * grid_y_size + c_y) * grid_x_size + c_x; 
    int dense_voxel_id = svid_to_dvid_map[sparse_voxel_id]; 

    #ifndef NDEBUG
    if (dense_voxel_id < 0)
    {
      printf("dense_voxel_id出现小于0的情况, 检查代码!");
      return;
    }
     #endif

    pid_to_dvid_map[p_id] = dense_voxel_id;

    const float* points_offset = dev_points + p_id * num_point_feature;
    atomicAdd(&voxel_count_list[dense_voxel_id], 1); // voxel_count_list 全初始化为 0
    // TODO: voxel 聚合可以 warp 优化（进阶）
    atomicAdd(&voxel_feats[dense_voxel_id * 3 + 0], points_offset[0]);
    atomicAdd(&voxel_feats[dense_voxel_id * 3 + 1], points_offset[1]);
    atomicAdd(&voxel_feats[dense_voxel_id * 3 + 2], points_offset[2]);
}

__global__ void voxel_mean_kernel(
  float* voxel_feats,
  const int* voxel_count_list,
  int* num_voxels)
{
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_voxels[0]) return;

  int cnt = voxel_count_list[v];
  // if (cnt <= 0) return;

  float inv = 1.0f / cnt;

  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    voxel_feats[v * 3 + c] = voxel_feats[v * 3 + c] * inv;
  }
}

__global__ void gather_point_feature_kernel(
  float* dev_pfe_gather_feature_,
  const float* voxel_means,
  const int* pid_to_dvid_map_,
  int num_gather_feature,
  int num_points)
{
  int p_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (p_id >= num_points) return;

  int dvid = pid_to_dvid_map_[p_id];
  if (dvid < 0) return;

  const float* mean = voxel_means + 3 * dvid;
  float* feat = dev_pfe_gather_feature_ + p_id * num_gather_feature;

  feat[4] -= mean[0];
  feat[5] -= mean[1];
  feat[6] -= mean[2];
  // #ifndef NDEBUG
  // if (p_id == 353023)
  // {
  //   printf("feat[0] = %.3f, feat[1] = %.3f, feat[2] = %.3f, feat[3] = %.3f,  feat[4] = %.3f, feat[5] = %.3f, feat[6] = %.3f, feat[9] = %.3f \n", 
  //     feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[9]);
  //   return;
  // }
  //  #endif

}

PreprocessPointsCuda::PreprocessPointsCuda(
    const int num_threads, const int max_num_pillars, const int max_num_points, const int max_points_per_pillar, 
    const int num_point_feature, const int kNumGatherPointFeature, const int num_inds_for_scan, 
    const int grid_x_size, const int grid_y_size, const int grid_z_size, 
    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size, 
    const float min_x_range, const float min_y_range, const float min_z_range)
    : num_threads_(num_threads),
      max_num_pillars_(max_num_pillars),
      max_num_points_(max_num_points),
      max_num_points_per_pillar_(max_points_per_pillar),
      num_point_feature_(num_point_feature),
      num_gather_point_feature_(kNumGatherPointFeature),
      num_inds_for_scan_(num_inds_for_scan),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
    
    // dynamic
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&coors_), max_num_points_ * 3 * sizeof(int)));  
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&voxel_num_), sizeof(int)));  
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&svid_to_dvid_map_), grid_x_size_ * grid_y_size_ * grid_z_size_ * sizeof(int)));  
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&voxel_feats_), max_num_pillars_ * 3 * sizeof(float)));  
    num_voxel_block_ = DIVUP(max_num_pillars_, num_threads_);    

    }

PreprocessPointsCuda::~PreprocessPointsCuda() {
    // dynamic
    GPU_CHECK(cudaFree(coors_));  
    GPU_CHECK(cudaFree(voxel_num_));  
    GPU_CHECK(cudaFree(svid_to_dvid_map_));  
    GPU_CHECK(cudaFree(voxel_feats_));  

  }


void PreprocessPointsCuda::DoPreprocessPointsCuda(
  const float* dev_points, const int in_num_points, void** buffer, cudaStream_t stream) {
  // initialize paraments
  float* dev_pfe_gather_feature = reinterpret_cast<float*>(buffer[0]);
  int* voxel_count_list_ = reinterpret_cast<int*>(buffer[1]);
  int* pid_to_dvid_map_ = reinterpret_cast<int*>(buffer[2]);
  int* voxel_coors_ = reinterpret_cast<int*>(buffer[3]);

  GPU_CHECK(cudaMemset(voxel_num_, 0,  sizeof(int)));
  GPU_CHECK(cudaMemset(svid_to_dvid_map_, -1,  grid_x_size_ * grid_y_size_ * grid_z_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(voxel_feats_, 0,  max_num_pillars_ * 3 * sizeof(float)));

  int num_point_block = DIVUP(in_num_points , num_threads_);
  dynamic_voxelize_step1_kernel<<<num_point_block , num_threads_, 0, stream>>>(
      dev_points, coors_, pillar_x_size_, pillar_y_size_, pillar_z_size_, 
      min_x_range_, min_y_range_, min_z_range_, grid_x_size_, grid_y_size_, grid_z_size_, in_num_points,
      num_point_feature_, voxel_num_, svid_to_dvid_map_, num_gather_point_feature_, dev_pfe_gather_feature, voxel_coors_);
      
  dynamic_voxelize_step2_kernel<<<num_point_block , num_threads_, 0, stream>>>(
      dev_points, coors_, voxel_feats_, voxel_count_list_, in_num_points, 
      num_point_feature_, pid_to_dvid_map_, svid_to_dvid_map_, grid_x_size_, grid_y_size_);
  // pid_to_dvid_map得到的 dvid 可能小于1
  voxel_mean_kernel<<<num_voxel_block_, num_threads_, 0, stream>>>(voxel_feats_, voxel_count_list_, voxel_num_);
  gather_point_feature_kernel<<<num_point_block, num_threads_, 0, stream>>>(dev_pfe_gather_feature, voxel_feats_, 
                                              pid_to_dvid_map_, num_gather_point_feature_, in_num_points);

}


