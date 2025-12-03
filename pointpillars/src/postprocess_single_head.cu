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


// headers in CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cmath>  // for expf
#include <cuda_runtime.h>

// headers in local files
#include "common.h"
#include "postprocess.h"
#include "check.h"
#include "bbox.h"
#include <stdio.h>

const int NMS_THREADS_PER_BLOCK = sizeof(uint64_t) * 8;
const int DET_CHANNEL = 9;

typedef struct {
  float val[DET_CHANNEL];
} combined_float;

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void postprocess_kernal(const float* cls_input,
                                   float* box_input,
                                   const float* dir_input,
                                   const float* anchors,
                                   const float* rotations,
                                   float* bndbox_output,
                                   float* score_output,
                                   int* object_counter,
                                   const float min_x_range,
                                   const float min_y_range,
                                   const float anchor_bottom_,
                                   const float resolution_x,
                                   const float resolution_y,
                                   const int feature_x_size,
                                   const int feature_y_size,
                                   const int num_anchors,
                                   const int num_classes,
                                   const int num_box_values,
                                   const float score_thresh,
                                   const float dir_offset)
{
  int loc_index = blockIdx.x;
  int ith_anchor = threadIdx.x;
  // 定位类别
  int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
  const float *scores = cls_input + cls_offset;

  float max_logit = scores[0];
  int cls_id = 0;
  for (int i = 1; i < num_classes; ++i) {
      float logit = scores[i];
      if (logit > max_logit) {
          max_logit = logit;
          cls_id = i;
      }
  }
  float max_score = sigmoid(max_logit);
  // mmdet版本的实现中用了topk，对于GPU来说topk成本高，不是必须
  if (max_score >= score_thresh)
  {
    
    // OPTIMIZE:这里把rotations的size定死了，若改变rotation的size，则失效
    const float *anchor_ptr = anchors + (ith_anchor >> 1) * 3; // 3为prior_w, prior_l, prior_h

    int col = loc_index % feature_x_size;
    int row = loc_index / feature_x_size;
    // 当前anchor的prior
    float x_offset = min_x_range + (col + 0.5) * resolution_x;
    float y_offset = min_y_range + (row + 0.5) * resolution_y;
    float z_offset = anchor_bottom_;

    // 定位bbox_pred和dir_cls_pred
    int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;

    float *box_encodings = box_input + box_offset;
    float wa = anchor_ptr[0];
    float la = anchor_ptr[1];
    float ha = anchor_ptr[2];
    float ra = rotations[ith_anchor & 1];
    float xa = x_offset;
    float ya = y_offset;
    float za = z_offset + ha / 2;
    float diagonal = sqrtf(wa * wa + la * la);

    box_encodings[3] = expf(box_encodings[3]) * wa;
    box_encodings[4] = expf(box_encodings[4]) * la;
    box_encodings[5] = expf(box_encodings[5]) * ha;
    box_encodings[6] = box_encodings[6] + ra;
    
    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * ha + za - box_encodings[5] / 2;

    int dir_label = dir_input[dir_cls_offset] > dir_input[dir_cls_offset + 1] ? 0 : 1;
    float val = box_encodings[6] - dir_offset;
    // Hack: 为什么这样做？
    const float period = 2 * M_PI / 2;
    float dir_rot = val - floor(val / (period + 1e-8) + 0.f) * period;
    float yaw = dir_rot + dir_offset + period * dir_label;

    int resCount = (int)atomicAdd(object_counter, 1);
    float *data = bndbox_output + resCount * DET_CHANNEL;
    data[0] = box_encodings[0];
    data[1] = box_encodings[1];
    data[2] = box_encodings[2];
    data[3] = box_encodings[3];
    data[4] = box_encodings[4];
    data[5] = box_encodings[5];
    data[6] = yaw;
    *(int *)&data[7] = cls_id;
    data[8] = max_score;
    score_output[resCount] = max_score;
    #ifndef NDEBUG
    printf("第 %d 个 anchor 的 data  = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", loc_index * num_anchors + ith_anchor, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    #endif
  }
}

__global__ void postprocess_kernal_long(const float* cls_input,
                                   float* box_input,
                                   const float* dir_input,
                                   const float* anchors,
                                   const float* rotations,
                                   float* bndbox_output,
                                   float* score_output,
                                   int* object_counter,
                                   const float min_x_range,
                                   const float min_y_range,
                                   const float anchor_bottom_,
                                   float resolution_x,
                                   float resolution_y,
                                   int feature_x_size,
                                   int feature_y_size,
                                   const int num_anchors,
                                   const int num_classes,
                                   const int num_box_values,
                                   const float score_thresh,
                                   const float dir_offset)
{
  int loc_index = blockIdx.x;
  int ith_anchor = threadIdx.x;
  // 定位类别
  int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
  const float *scores = cls_input + cls_offset;

  float max_logit = scores[0];
  int cls_id = 0;
  for (int i = 1; i < num_classes; ++i) {
      float logit = scores[i];
      if (logit > max_logit) {
          max_logit = logit;
          cls_id = i;
      }
  }
  float max_score = sigmoid(max_logit);
  // mmdet版本的实现中用了topk，对于GPU来说topk成本高，不是必须
  if (max_score >= score_thresh)
  {
    
    // OPTIMIZE:这里把rotations的size定死了，若改变rotation的size，则失效
    const float *anchor_ptr = anchors + (ith_anchor >> 1) * 3; // 3为prior_w, prior_l, prior_h
    float wa = anchor_ptr[0];
    float la = anchor_ptr[1];
    float ha = anchor_ptr[2];

    // 定位bbox_pred和dir_cls_pred
    int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;

    while (loc_index >= feature_x_size * feature_y_size){
        loc_index -= feature_x_size * feature_y_size;
        feature_x_size >>= 1;
        feature_y_size >>= 1;    
        resolution_x *= 2;
        resolution_y *= 2;
        wa *= 2;
        la *= 2;
        ha *= 2;
    }

    int col = loc_index % feature_x_size;
    int row = loc_index / feature_x_size;
    // 当前anchor的prior
    float x_offset = min_x_range + (col + 0.5) * resolution_x;
    float y_offset = min_y_range + (row + 0.5) * resolution_y;
    float z_offset = anchor_bottom_;

    float *box_encodings = box_input + box_offset;
    float ra = rotations[ith_anchor & 1];
    float xa = x_offset;
    float ya = y_offset;
    float za = z_offset + ha / 2;
    #ifndef NDEBUG
    if ( loc_index * num_anchors + ith_anchor == 30841)
    printf("xa = %.3f, ya = %.3f, za = %.3f wa = %.3f, la = %.3f, ha = %.3f, ra = %.3f \n", xa, ya, za, wa, la, ha, ra);
    #endif

    float diagonal = sqrtf(wa * wa + la * la);

    box_encodings[3] = expf(box_encodings[3]) * wa;
    box_encodings[4] = expf(box_encodings[4]) * la;
    box_encodings[5] = expf(box_encodings[5]) * ha;
    box_encodings[6] = box_encodings[6] + ra;

    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * ha + za - box_encodings[5] / 2;

    int dir_label = dir_input[dir_cls_offset] > dir_input[dir_cls_offset + 1] ? 0 : 1;
    float val = box_encodings[6] - dir_offset;
    // Hack: 为什么这样做？
    const float period = 2 * M_PI / 2;
    float dir_rot = val - floor(val / (period + 1e-8) + 0.f) * period;
    float yaw = dir_rot + dir_offset + period * dir_label;

    int resCount = (int)atomicAdd(object_counter, 1);
    float *data = bndbox_output + resCount * DET_CHANNEL;
    data[0] = box_encodings[0];
    data[1] = box_encodings[1];
    data[2] = box_encodings[2];
    data[3] = box_encodings[3];
    data[4] = box_encodings[4];
    data[5] = box_encodings[5];
    data[6] = yaw;
    *(int *)&data[7] = cls_id;
    data[8] = max_score;
    score_output[resCount] = max_score;
    #ifndef NDEBUG
    if ( loc_index * num_anchors + ith_anchor == 30841)
    printf("第 %d 个 anchor 的 data  = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", loc_index * num_anchors + ith_anchor, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    #endif
  }
}

cudaError_t postprocess_launch(const float *cls_input,
                      float *box_input,
                      const float *dir_input,
                      const float *anchors,
                      const float *rotations,
                      float *bndbox_output,
                      float *score_output,
                      int *object_counter,
                      const float min_x_range,
                      const float min_y_range,
                      const float anchor_bottom,
                      const float resolution_x,
                      const float resolution_y,
                      const int feature_x_size,
                      const int feature_y_size,
                      const int num_anchors,
                      const int num_classes,
                      const int num_feature,
                      const int num_box_values,
                      const float score_thresh,
                      const float dir_offset,
                      cudaStream_t stream)
{
  dim3 threads (num_anchors);
  dim3 blocks (num_feature);

  postprocess_kernal_long<<<blocks, threads, 0, stream>>>(
      cls_input, box_input, dir_input, anchors, rotations, bndbox_output,
      score_output, object_counter, min_x_range, min_y_range, anchor_bottom,
      resolution_x, resolution_y, feature_x_size, feature_y_size, num_anchors,
      num_classes, num_box_values, score_thresh, dir_offset);
  return cudaGetLastError();
}

__device__ inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ inline int check_box2d(float const *const box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box[0];
    float center_y = box[1];
    float angle_cos = cos(-box[6]);
    float angle_sin = sin(-box[6]);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( fmin(p0.x, p1.x) <= fmax(q0.x, q1.x) &&
          fmin(q0.x, q1.x) <= fmax(p0.x, p1.x) &&
          fmin(p0.y, p1.y) <= fmax(q0.y, q1.y) &&
          fmin(q0.y, q1.y) <= fmax(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > 1e-8) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

__device__ inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

__device__ inline bool devIoU(float const *const box_a, float const *const box_b, const float nms_thresh) {
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {box_a[0], box_a[1]};
    float2 center_b = float2 {box_b[0], box_b[1]};

    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }

    float s_overlap = fabs(area) / 2.0;;
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float iou = s_overlap / fmaxf(sa + sb - s_overlap, 1e-8);

    return iou >= nms_thresh;
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold, const float *dev_boxes, uint64_t *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int tid = threadIdx.x;

  if (row_start > col_start) return;

  const int row_size = fminf(n_boxes - row_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
  const int col_size = fminf(n_boxes - col_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);

  __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * 7];

  if (tid < col_size) {
    int idx = NMS_THREADS_PER_BLOCK * col_start + tid;
    block_boxes[tid * 7 + 0] = dev_boxes[idx * DET_CHANNEL + 0];
    block_boxes[tid * 7 + 1] = dev_boxes[idx * DET_CHANNEL + 1];
    block_boxes[tid * 7 + 2] = dev_boxes[idx * DET_CHANNEL + 2];
    block_boxes[tid * 7 + 3] = dev_boxes[idx * DET_CHANNEL + 3];
    block_boxes[tid * 7 + 4] = dev_boxes[idx * DET_CHANNEL + 4];
    block_boxes[tid * 7 + 5] = dev_boxes[idx * DET_CHANNEL + 5];
    block_boxes[tid * 7 + 6] = dev_boxes[idx * DET_CHANNEL + 6];
  }
  __syncthreads();

  if (tid < row_size) {
    const int cur_box_idx = NMS_THREADS_PER_BLOCK * row_start + tid;
    const float *cur_box = dev_boxes + cur_box_idx * DET_CHANNEL;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    dev_mask[cur_box_idx * gridDim.y + col_start] = t;
  }
}

cudaError_t nms_launch(unsigned int boxes_num,
    float *boxes,
    float nms_thresh,
    uint64_t* mask,
    cudaStream_t stream)
{
int col_blocks = DIVUP(boxes_num, NMS_THREADS_PER_BLOCK);

dim3 blocks(col_blocks, col_blocks);
dim3 threads(NMS_THREADS_PER_BLOCK);

nms_cuda<<<blocks, threads, 0, stream>>>(boxes_num, nms_thresh, boxes, mask);

return cudaGetLastError();
}
  

void PostprocessSingleHead::DoPostprocess(const float* cls, const float* box, const float* dir, void* stream)
{
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemsetAsync(object_counter_, 0, sizeof(int), _stream));
    // WTF
    // checkRuntime(cudaMemsetAsync(h_mask_, 0, h_mask_size_, _stream));

    checkRuntime(postprocess_launch((float *)cls,
                                    (float *)box,
                                    (float *)dir,
                                    anchors_dev_,
                                    rotations_dev_,
                                    bndbox_,
                                    score_,
                                    object_counter_,
                                    min_range_.x,
                                    min_range_.y,
                                    anchor_bottom_,
                                    resolution_.x,
                                    resolution_.y,
                                    feature_size_[0].x,
                                    feature_size_[0].y,
                                    num_anchors_,
                                    num_classes_,
                                    num_features_,
                                    num_box_values_,
                                    score_thresh_,
                                    dir_offset_,
                                    _stream));
    checkRuntime(cudaMemcpy(&bndbox_num_, object_counter_, sizeof(int), cudaMemcpyDeviceToHost));

    //在 GPU 上，将 bndbox_ 检测框数组按置信度 score_ 降序排序，保持框与分数一一对应
    thrust::device_ptr<combined_float> thr_bndbox_((combined_float *)bndbox_);
    thrust::sort_by_key(thrust::cuda::par.on(_stream), score_, score_ + bndbox_num_, thr_bndbox_, thrust::greater<float>());

    int nms_pre_maxsize_ = 500; //TODO 参数化
    bndbox_num_ = min(nms_pre_maxsize_, bndbox_num_);
    uint64_t *dev_mask = nullptr;
    unsigned int mask_size = bndbox_num_ * DIVUP(bndbox_num_, NMS_THREADS_PER_BLOCK) * sizeof(uint64_t);
    GPU_CHECK(cudaMalloc(&dev_mask, mask_size));    


    checkRuntime(nms_launch(bndbox_num_, bndbox_, nms_thresh_, dev_mask, _stream));
    checkRuntime(cudaMemcpy(h_bndbox_, bndbox_, bndbox_num_ * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    uint64_t* h_mask_ = (uint64_t*)malloc(mask_size);
    GPU_CHECK(cudaMemcpy(h_mask_, dev_mask, mask_size, cudaMemcpyDeviceToHost));
    int col_blocks = DIVUP(bndbox_num_, NMS_THREADS_PER_BLOCK);
    memset(remv_.data(), 0, col_blocks * sizeof(uint64_t));
    bndbox_num_after_nms_ = 0;
    
    for (unsigned int i_nms = 0; i_nms < bndbox_num_; i_nms++) {
        unsigned int nblock = i_nms / NMS_THREADS_PER_BLOCK;
        unsigned int inblock = i_nms % NMS_THREADS_PER_BLOCK;

        if (!(remv_[nblock] & (1ULL << inblock))) {
            bndbox_after_nms_[bndbox_num_after_nms_] = *(BoundingBox*)(&h_bndbox_[i_nms * 9]);
            bndbox_num_after_nms_++;
            uint64_t* p = h_mask_ + i_nms * col_blocks;
            for (int j_nms = nblock; j_nms < col_blocks; j_nms++) {
                remv_[j_nms] |= p[j_nms];
            }
        }
    }
}

PostprocessSingleHead::~PostprocessSingleHead() {
    if (bndbox_) checkRuntime(cudaFree(bndbox_));
    if (h_bndbox_) checkRuntime(cudaFreeHost(h_bndbox_));
    if (score_) checkRuntime(cudaFree(score_));

    if (anchors_dev_) checkRuntime(cudaFree(anchors_dev_));
    if (rotations_dev_) checkRuntime(cudaFree(rotations_dev_));
    if (object_counter_) checkRuntime(cudaFree(object_counter_));

    // if (h_mask_) checkRuntime(cudaFreeHost(h_mask_));
}

PostprocessSingleHead::PostprocessSingleHead(
        const nvtype::Float3 min_range,
        const nvtype::Float3 max_range,
        const std::vector<nvtype::Int2> feature_size,
        const int num_classes,
        const int num_features,
        std::vector<std::vector<float>> anchors,
        float anchor_bottom,
        std::vector<float> rotations,
        const int num_box_values,
        const float score_thresh,
        const float dir_offset,
        const float nms_thresh)
        : min_range_(min_range), max_range_(max_range), feature_size_(feature_size), anchor_bottom_(anchor_bottom),
        num_classes_(num_classes), num_features_(num_features), anchors_host_(anchors), rotations_host_(rotations), 
        num_box_values_(num_box_values), score_thresh_(score_thresh), dir_offset_(dir_offset), nms_thresh_(nms_thresh)
{
    if (rotations_host_.size() != 2) {
        throw std::runtime_error("目前只支持rotation_.size为 2 的实现");
    }
    int n_anchor_size = anchors_host_.size();
    int n_rotation = rotations_host_.size();
    num_anchors_ = n_anchor_size * n_rotation;
    num_det_ = num_features_ * num_anchors_;
    resolution_.x = (max_range_.x - min_range_.x) /  feature_size_[0].x;
    resolution_.y = (max_range_.y - min_range_.y) /  feature_size_[0].y;

    int len_per_anchor = anchors_host_[0].size();
    GPU_CHECK(cudaMalloc((void **)&bndbox_, num_det_ * num_box_values_ * sizeof(float)));
    GPU_CHECK(cudaMallocHost((void **)&h_bndbox_, num_det_ * num_box_values_ * sizeof(float)));
    GPU_CHECK(cudaMalloc((void **)&score_, num_det_ * sizeof(float)));

    GPU_CHECK(cudaMalloc((void **)&anchors_dev_, n_anchor_size * len_per_anchor * sizeof(float)));
    GPU_CHECK(cudaMalloc((void **)&rotations_dev_, n_rotation * sizeof(float)));

    GPU_CHECK(cudaMalloc((void **)&object_counter_, sizeof(int)));

    std::vector<float> anchors_flat(n_anchor_size * len_per_anchor);
    for (int i = 0; i < n_anchor_size; ++i) {
        memcpy(anchors_flat.data() + i * len_per_anchor,
        anchors_host_[i].data(),
               len_per_anchor * sizeof(float));
    }
    GPU_CHECK(cudaMemcpy(anchors_dev_, anchors_flat.data(), n_anchor_size * len_per_anchor * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(rotations_dev_, rotations_host_.data(), n_rotation * sizeof(float), cudaMemcpyHostToDevice));

    // WTF: h_mask_size_作用？
    // h_mask_size_ = num_det_ * DIVUP(num_det_, NMS_THREADS_PER_BLOCK) * sizeof(uint64_t);
    // GPU_CHECK(cudaMallocHost((void **)&h_mask_, h_mask_size_));

    int res_blocks = DIVUP(num_det_, NMS_THREADS_PER_BLOCK);
    remv_ = std::vector<uint64_t>(res_blocks, 0);
    bndbox_after_nms_.resize(num_det_);
}

std::vector<BoundingBox> PostprocessSingleHead::bndBoxVec(){
    return std::vector<BoundingBox>(bndbox_after_nms_.begin(), bndbox_after_nms_.begin() + bndbox_num_after_nms_);
}
