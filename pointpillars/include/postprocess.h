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

#pragma once
#include <memory>
#include <vector>
#include <stdexcept>
#include "nms.h"
#include "dtype.h"
#include "bbox.h"


/**
 * @brief 后处理基类，定义统一接口
 * @details 所有后处理实现类都应继承自此基类
 */
class Postprocess {
 public:
  virtual ~Postprocess() = default;

  /**
   * @brief 执行后处理（多头部版本重载）
   * @param[in] cls_pred_0 类别预测0
   * @param[in] cls_pred_12 类别预测1-2
   * @param[in] cls_pred_34 类别预测3-4
   * @param[in] cls_pred_5 类别预测5
   * @param[in] cls_pred_67 类别预测6-7
   * @param[in] cls_pred_89 类别预测8-9
   * @param[in] box_preds 边界框预测
   * @param[in] host_box 主机端边界框缓冲区
   * @param[in] host_score 主机端分数缓冲区
   * @param[in] host_filtered_count 主机端过滤计数
   * @param[out] out_detection 输出检测结果
   * @param[out] out_label 输出标签
   * @param[out] out_score 输出分数
   */
  virtual void DoPostprocess(
      float* cls_pred_0,
      float* cls_pred_12,
      float* cls_pred_34,
      float* cls_pred_5,
      float* cls_pred_67,
      float* cls_pred_89,
      const float* box_preds,
      float* host_box,
      float* host_score,
      int* host_filtered_count,
      std::vector<float>& out_detection,
      std::vector<int>& out_label,
      std::vector<float>& out_score) {
    // 默认实现：不支持多头部接口，抛出异常
    throw std::runtime_error("DoPostprocess: Multi-head interface not supported by this implementation");
  }

  /**
   * @brief 执行后处理（单头部版本重载）
   * @param[in] cls 类别预测
   * @param[in] box 边界框预测
   * @param[in] dir 方向预测
   * @param[in] stream CUDA流
   */
  virtual void DoPostprocess(
      const float* cls,
      const float* box,
      const float* dir,
      void* stream) {
    // 默认实现：不支持单头部接口，抛出异常
    throw std::runtime_error("DoPostprocess: Single-head interface not supported by this implementation");
  }
  virtual std::vector<BoundingBox> bndBoxVec(){
    throw std::runtime_error("bndBoxVec: Single-head interface not supported by this implementation");
  };
};

/**
 * @brief 多头部后处理实现类
 * @details 处理多个检测头输出的后处理
 */
class PostprocessMultiHead : public Postprocess {
 private:
  const int num_threads_;
  const float float_min_;
  const float float_max_;
  const int num_class_;
  const int num_anchor_per_cls_;
  const float score_threshold_;
  const float nms_overlap_threshold_;
  const int nms_pre_maxsize_;
  const int nms_post_maxsize_;
  const int num_box_corners_;
  const int num_input_box_feature_;
  const int num_output_box_feature_;
  const std::vector<std::vector<int>> multihead_label_mapping_;

  std::unique_ptr<NmsCuda> nms_cuda_ptr_;

 public:
  /**
   * @brief 构造函数
   * @param[in] num_threads CUDA核启动时的线程数
   * @param[in] float_min 最小浮点值
   * @param[in] float_max 最大浮点值
   * @param[in] num_class 类别数
   * @param[in] num_anchor_per_cls 每个类别的锚点数
   * @param[in] multihead_label_mapping 多头部标签映射
   * @param[in] score_threshold 分数阈值
   * @param[in] nms_overlap_threshold NMS的IOU阈值
   * @param[in] nms_pre_maxsize NMS前的最大框数
   * @param[in] nms_post_maxsize NMS后的最大框数
   * @param[in] num_box_corners 边界框角点数
   * @param[in] num_input_box_feature 输入边界框特征数
   * @param[in] num_output_box_feature 输出边界框特征数
   */
  PostprocessMultiHead(
      const int num_threads,
      const float float_min,
      const float float_max,
      const int num_class,
      const int num_anchor_per_cls,
      const std::vector<std::vector<int>> multihead_label_mapping,
      const float score_threshold,
      const float nms_overlap_threshold,
      const int nms_pre_maxsize,
      const int nms_post_maxsize,
      const int num_box_corners,
      const int num_input_box_feature,
      const int num_output_box_feature);

  virtual ~PostprocessMultiHead() = default;

  /**
  using Postprocess::DoPostprocess; 的作用是：
  1. 显式引入基类的所有同名重载版本到派生类的作用域
  2. 明确告知编译器：这是有意使用基类的实现，而非遗漏
  3. 消除歧义：让编译器知道派生类“看到”了所有重载版本
  */
  using Postprocess::DoPostprocess;

  /**
   * @brief 执行多头部后处理
   */
  void DoPostprocess(
      float* cls_pred_0,
      float* cls_pred_12,
      float* cls_pred_34,
      float* cls_pred_5,
      float* cls_pred_67,
      float* cls_pred_89,
      const float* box_preds,
      float* host_box,
      float* host_score,
      int* host_filtered_count,
      std::vector<float>& out_detection,
      std::vector<int>& out_label,
      std::vector<float>& out_score) override;
};


/**
 * @brief 单头部后处理实现类
 * @details 处理单个检测头输出的后处理
 */
class PostprocessSingleHead : public Postprocess {
 private:
  const nvtype::Float3 min_range_;
  const nvtype::Float3 max_range_;
  nvtype::Float2 resolution_;

  std::vector<nvtype::Int2> feature_size_;
  int num_classes_;
  int num_anchors_;
  int num_features_;
  int num_box_values_;
  float score_thresh_;
  float dir_offset_;
  float nms_thresh_;

  std::vector<std::vector<float>> anchors_host_;
  std::vector<float> rotations_host_;

  // int num_classes_ = 10;
  // int num_anchors_ = 8;
  // float anchors_[12] = {
  //     2.5981, 0.8660, 1.,
  //     1.7321, 0.5774, 1.,
  //     1., 1., 1.,
  //     0.4, 0.4, 1.};
  // int num_box_values_ = 9;
  // float score_thresh_ = 0.1;
  // float dir_offset_ = 0.7854;
  // float nms_thresh_ = 0.01;

  float* anchors_dev_ = nullptr;
  float* rotations_dev_ = nullptr;
  float anchor_bottom_;
  int* object_counter_ = nullptr;

  float* bndbox_ = nullptr;
  float* h_bndbox_ = nullptr;
  float* score_ = nullptr;
  unsigned int num_det_ = 0;
  // WTF
  // uint64_t* h_mask_ = nullptr;
  // unsigned int h_mask_size_ = 0;
  std::vector<uint64_t> remv_;

  unsigned int bndbox_num_ = 0;
  std::vector<BoundingBox> bndbox_after_nms_;
  unsigned int bndbox_num_after_nms_ = 0;

 public:
  /**
   * @brief 构造函数
   * @param[in] 参数待定义
   */
  PostprocessSingleHead(
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
    const float nms_thresh);

  ~PostprocessSingleHead() override;

  /**
  using Postprocess::DoPostprocess; 的作用是：
  1. 显式引入基类的所有同名重载版本到派生类的作用域
  2. 明确告知编译器：这是有意使用基类的实现，而非遗漏
  3. 消除歧义：让编译器知道派生类“看到”了所有重载版本
  */
  using Postprocess::DoPostprocess;

  /**
   * @brief 执行后处理（单头部版本重载）
   */
  void DoPostprocess(
      const float* cls,
      const float* box,
      const float* dir,
      void* stream) override;

  std::vector<BoundingBox> bndBoxVec();

};
  