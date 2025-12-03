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

// headers in STL
// 标准模板库头文件
#include <algorithm>  // 算法库，提供sort、max、min等常用算法
#include <cmath>      // 数学函数库，提供sin、cos、sqrt等数学运算
#include <iomanip>    // 输入输出流格式控制库，用于格式化输出
#include <limits>     // 数值极限库，提供数值类型的最大值、最小值等
#include <map>        // 映射容器库，提供键值对映射数据结构
#include <memory>     // 智能指针库，提供shared_ptr、unique_ptr等
#include <string>     // 字符串库，提供string类和相关操作
#include <vector>     // 向量容器库，提供动态数组数据结构
#include <iostream>   // 输入输出流库，提供cin、cout等标准I/O
#include <sstream>    // 字符串流库，提供stringstream用于字符串处理
#include <fstream>    // 文件流库，提供文件读写功能

// headers in TensorRT
// TensorRT推理引擎头文件
#include "NvInfer.h"        // TensorRT核心库，提供ICudaEngine、IExecutionContext等接口
#include "NvOnnxParser.h"   // ONNX解析器库，用于将ONNX模型转换为TensorRT引擎

// headers in local files
// 本地文件头文件
// #include "params.h"          // 参数头文件（已注释，不再使用）
#include "common.h"          // 通用定义和工具函数头文件
#include <yaml-cpp/yaml.h>   // YAML配置文件解析库，用于读取配置文件
#include "preprocess.h"      // 预处理模块头文件，包含点云预处理相关类
#include "scatter.h"         // 散射操作模块头文件，包含特征散射相关类
#include "postprocess.h"     // 后处理模块头文件，包含检测结果后处理相关类

using namespace std;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
      : reportable_severity(severity) {}

  void log(Severity severity, const char* msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;
};



class PointPillars {
 private:
    // initialize in initializer list
    const float score_threshold_;
    const float nms_overlap_threshold_;
    const bool use_onnx_;
    const std::string pfe_file_;
    const std::string backbone_file_;
    const std::string pp_config_;
    // end initializer list
    // voxel size
    float kPillarXSize;
    float kPillarYSize;
    float kPillarZSize;
    // point cloud range
    float kMinXRange;
    float kMinYRange;
    float kMinZRange;
    float kMaxXRange;
    float kMaxYRange;
    float kMaxZRange;
    // hyper parameters
    int kNumClass;
    int kMaxNumPillars;
    int kMaxNumPointsPerPillar;
    int kNumPointFeature;
    int kNumGatherPointFeature;
    int kGridXSize;
    int kGridYSize;
    int kGridZSize;
    int kNumAnchorXinds;
    int kNumAnchorYinds;
    int kRpnInputSize;
    int kNumAnchor;
    int kNumFeature;
    int kNumInputBoxFeature;
    int kNumOutputBoxFeature;
    int kRpnBoxOutputSize;
    int kRpnClsOutputSize;
    int kRpnDirOutputSize;
    int kBatchSize;
    int kNumIndsForScan;
    int kNumThreads; // TODO: 与pfe输出的channel数有关，需要解耦？
    // if you change kNumThreads, need to modify NUM_THREADS_MACRO in
    // common.h
    int kNumBoxCorners;
    int kNmsPreMaxsize;
    int kNmsPostMaxsize;
    //params for initialize anchors
    //Adapt to MMdet
    std::vector<int> kLayerStrides;
    std::vector<nvtype::Int2> kFeatureSize;
    //Adapt to OpenPCDet
    int kAnchorStrides;
    float kDirOffset;
    std::vector<string> kAnchorNames;
    std::vector<float> kAnchorDxSizes;
    std::vector<float> kAnchorDySizes;
    std::vector<float> kAnchorDzSizes;

    std::vector<std::vector<float>> kAnchorSizes;
    float kAnchorBottom;
    std::vector<float> kAnchorRotations;
    int kLenPerAnchor;

    std::vector<std::vector<int>> kMultiheadLabelMapping;
    int kNumAnchorPerCls;
    int host_pillar_count_[1];

    int* dev_x_coors_;
    int* dev_y_coors_;
    float* dev_num_points_per_pillar_;
    int* dev_sparse_pillar_map_;
    int* dev_cumsum_along_x_;
    int* dev_cumsum_along_y_;

    float* dev_pillar_point_feature_;
    float* dev_pillar_coors_;
    float* dev_points_mean_;

    float* dev_pfe_gather_feature_;
    void* pfe_buffers_[2];
    //variable for doPostprocessCudaMultiHead
    void* rpn_buffers_[4];
    
    std::vector<float*> rpn_box_output_; 
    std::vector<float*> rpn_cls_output_;

    float* dev_scattered_feature_;

    float* host_box_;
    float* host_score_;
    int*   host_filtered_count_;

    // float* dev_filtered_box_;
    // float* dev_filtered_score_;
    // int*   dev_filtered_label_;
    // int*   dev_filtered_dir_;
    // float* dev_box_for_nms_;
    // int*   dev_filter_count_;

    std::unique_ptr<PreprocessPointsCuda> preprocess_points_cuda_ptr_;
    std::unique_ptr<ScatterCuda> scatter_cuda_ptr_;
    std::unique_ptr<Postprocess> postprocess_ptr_;

    Logger g_logger_;
    nvinfer1::ICudaEngine* pfe_engine_;
    nvinfer1::ICudaEngine* backbone_engine_;
    nvinfer1::IExecutionContext* pfe_context_;
    nvinfer1::IExecutionContext* backbone_context_;

    /**
     * @brief Memory allocation for device memory
     * @details Called in the constructor
     */
    void DeviceMemoryMalloc();

    /**
     * @brief Memory set to 0 for device memory
     * @details Called in the DoInference
     */
    void SetDeviceMemoryToZero();

    /**
     * @brief Initializing paraments from pointpillars.yaml
     * @details Called in the constructor
     */
    void InitParams();
    /**
     * @brief Initializing TensorRT instances
     * @param[in] usr_onnx_ if true, parse ONNX 
     * @details Called in the constructor
     */
    void InitTRT(const bool use_onnx);
    /**
     * @brief Convert ONNX to TensorRT model
     * @param[in] model_file ONNX model file path
     * @param[out] engine_ptr TensorRT model engine made out of ONNX model
     * @details Load ONNX model, and convert it to TensorRT model
     */
    void OnnxToTRTModel(const std::string& model_file,
                        nvinfer1::ICudaEngine** engine_ptr);

    /**
     * @brief Convert Engine to TensorRT model
     * @param[in] model_file Engine(TensorRT) model file path
     * @param[out] engine_ptr TensorRT model engine made 
     * @details Load Engine model, and convert it to TensorRT model
     */
    void EngineToTRTModel(const std::string &engine_file ,     
                        nvinfer1::ICudaEngine** engine_ptr) ;

    /**
     * @brief Preproces points
     * @param[in] in_points_array Point cloud array
     * @param[in] in_num_points Number of points
     * @details Call CPU or GPU preprocess
     */
    void Preprocess(const float* in_points_array, const int in_num_points);

    public:
    /**
     * @brief Constructor
     * @param[in] score_threshold Score threshold for filtering output
     * @param[in] nms_overlap_threshold IOU threshold for NMS
     * @param[in] use_onnx if true,using onnx file ,else using engine file
     * @param[in] pfe_file Pillar Feature Extractor ONNX file path
     * @param[in] rpn_file Region Proposal Network ONNX file path
     * @details Variables could be changed through point_pillars_detection
     */
    PointPillars(const float score_threshold,
                const float nms_overlap_threshold,
                const bool use_onnx,
                const std::string pfe_file,
                const std::string rpn_file,
                const std::string pp_config);
    ~PointPillars();

    /**
     * @brief Call PointPillars for the inference
     * @param[in] in_points_array Point cloud array
     * @param[in] in_num_points Number of points
     * @param[out] out_detections Network output bounding box
     * @param[out] out_labels Network output object's label
     * @details This is an interface for the algorithm
     */
    std::vector<BoundingBox> DoInference(const float* in_points_array,
                    const int in_num_points,
                    std::vector<float>* out_detections,
                    std::vector<int>* out_labels,
                    std::vector<float>* out_scores);
};

