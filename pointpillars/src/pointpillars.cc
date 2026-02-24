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


#include "pointpillars.h"

#include <iostream>
#include <dlfcn.h>

void PointPillars::LoadTRTPlugins() {
    void* handle = dlopen(
        "/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/build/libpointpillars_trt_plugins.so",
        RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "dlopen plugin failed: " << dlerror() << std::endl;
    } else {
        std::cout << "Plugin loaded OK" << std::endl;
    }
}
void PointPillars::InitParams()
{
    YAML::Node params = YAML::LoadFile(pp_config_);
    kPillarXSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
    kPillarYSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
    kPillarZSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
    kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
    kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
    kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
    kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
    kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
    kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
    kNumClass = params["CLASS_NAMES"].size();
    kMaxNumPillars = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"].as<int>();
    kMaxNumPointsPerPillar = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"].as<int>();
    kMaxNumPoints = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["kMaxNumPoints"].as<int>();
    kNumPointFeature = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["kNumPointFeature"].as<int>();
    if (kNumPointFeature != 4) {
        throw std::runtime_error("目前只支持kNumPointFeature为 4 的实现");
    }
    kNumGatherPointFeature = kNumPointFeature;
    if (params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["with_cluster_center"].as<bool>())
        kNumGatherPointFeature += 3;
    if (params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["with_voxel_center"].as<bool>())
        kNumGatherPointFeature += 3;

    kNumInputBoxFeature = 7;
    kNumOutputBoxFeature = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["BOX_CODER_CONFIG"]["code_size"].as<int>();
    kBatchSize = 1;
    kNumIndsForScan = 1024;
    kNumThreads = 64;
    kNumBoxCorners = 8;
    kAnchorStrides = 4;
    kLayerStrides = params["MODEL"]["BACKBONE_2D"]["LAYER_STRIDES"].as<std::vector<int>>();
    kNmsPreMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"].as<int>();
    kNmsPostMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"].as<int>();
    //params for initialize anchors
    //Adapt to MMdet
    kDirOffset = params["MODEL"]["DENSE_HEAD"]["DIR_OFFSET"].as<float>();
    kAnchorSizes = params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"]["anchor_sizes"].as<std::vector<std::vector<float>>>();
    kAnchorBottom = params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"]["anchor_bottom_heights"].as<float>();
    kAnchorRotations = params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"]["anchor_rotations"].as<std::vector<float>>();
    for (int idx_head = 0; idx_head < params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"].size(); ++idx_head)
    {
        int num_cls_per_head = params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"][idx_head]["HEAD_CLS_NAME"].size();
        std::vector<int> value;
        for (int i = 0; i < num_cls_per_head; ++i)
        {
            value.emplace_back(idx_head + i);
        }
        kMultiheadLabelMapping.emplace_back(value);
    }


    // Generate secondary parameters based on above.
    kGridXSize = static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize); //512
    kGridYSize = static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize); //512
    kGridZSize = static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize); //1
    kRpnInputSize = 64 * kGridYSize * kGridXSize;

    kFeatureSize.clear();
    int stride = 1;
    kNumFeature = 0;
    for (auto s : kLayerStrides) {
        if (s != 2) {
            throw std::runtime_error("目前只支持 stride 为 2 的多尺度特征输出");
        }
        stride *= s;
        int fs_x = kGridXSize / stride, fs_y = kGridYSize / stride;
        kNumFeature += (fs_x * fs_y);
        kFeatureSize.emplace_back(fs_x, fs_y);
    }
        // kNumAnchorXinds = static_cast<int>(kGridXSize / kAnchorStrides); //Width
    // kNumAnchorYinds = static_cast<int>(kGridYSize / kAnchorStrides); //Hight
    // kNumAnchor = kNumAnchorXinds * kNumAnchorYinds * 2 * kNumClass;  // H * W * Ro * N = 196608

    // kNumAnchorPerCls = kNumAnchorXinds * kNumAnchorYinds * 2; //H * W * Ro = 32768
    // kRpnBoxOutputSize = kNumAnchor * kNumOutputBoxFeature;
    // kRpnClsOutputSize = kNumAnchor * kNumClass;
    // kRpnDirOutputSize = kNumAnchor * 2;
}


PointPillars::PointPillars(const float score_threshold,
                           const float nms_overlap_threshold,
                           const bool use_onnx,
                           const std::string pfe_file,
                           const std::string backbone_file,
                           const std::string pp_config)
    : score_threshold_(score_threshold),
      nms_overlap_threshold_(nms_overlap_threshold),
      use_onnx_(use_onnx),
      pfe_file_(pfe_file),
      backbone_file_(backbone_file),
      pp_config_(pp_config)
{
    InitParams();
    LoadTRTPlugins();
    InitTRT(use_onnx_);
    GPU_CHECK(cudaStreamCreate(&stream_));
    // Create CUDA events for timing
    GPU_CHECK(cudaEventCreate(&preprocess_start_));
    GPU_CHECK(cudaEventCreate(&preprocess_end_));
    GPU_CHECK(cudaEventCreate(&pfe_start_));
    GPU_CHECK(cudaEventCreate(&pfe_end_));
    GPU_CHECK(cudaEventCreate(&scatter_start_));
    GPU_CHECK(cudaEventCreate(&scatter_end_));
    GPU_CHECK(cudaEventCreate(&backbone_start_));
    GPU_CHECK(cudaEventCreate(&backbone_end_));
    GPU_CHECK(cudaEventCreate(&postprocess_start_));
    GPU_CHECK(cudaEventCreate(&postprocess_end_));
    DeviceMemoryMalloc();
    preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
        kNumThreads,
        kMaxNumPillars,
        kMaxNumPoints,
        kMaxNumPointsPerPillar,
        kNumPointFeature,
        kNumGatherPointFeature,
        kNumIndsForScan,
        kGridXSize,kGridYSize, kGridZSize,
        kPillarXSize,kPillarYSize, kPillarZSize,
        kMinXRange, kMinYRange, kMinZRange));


    scatter_cuda_ptr_.reset(new ScatterCuda(kNumThreads, kGridXSize, kGridYSize));

    // const float float_min = std::numeric_limits<float>::lowest();
    // const float float_max = std::numeric_limits<float>::max();
    postprocess_ptr_.reset(
        new PostprocessSingleHead(
            nvtype::Float3(kMinXRange, kMinYRange, kMinZRange),
            nvtype::Float3(kMaxXRange, kMaxYRange, kMaxZRange),
            kFeatureSize,
            kNumClass,
            kNumFeature,
            kAnchorSizes,
            kAnchorBottom,
            kAnchorRotations,
            kNumOutputBoxFeature,
            score_threshold_,
            kDirOffset,
            nms_overlap_threshold_
        )); 
        // postprocess_ptr_.reset(
        // new PostprocessMultiHead(kNumThreads,
        //                 float_min, float_max, 
        //                 kNumClass,kNumAnchorPerCls,
        //                 kMultiheadLabelMapping,
        //                 score_threshold_, 
        //                 nms_overlap_threshold_,
        //                 kNmsPreMaxsize, 
        //                 kNmsPostMaxsize,
        //                 kNumBoxCorners, 
        //                 kNumInputBoxFeature,
        //                 kNumOutputBoxFeature));  /*kNumOutputBoxFeature*/
}


void PointPillars::DeviceMemoryMalloc() {
    // for pillars 
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_), kMaxNumPoints * kNumPointFeature * sizeof(float)));

    // for backbone
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pfe_buffers_[0]), kMaxNumPoints * kNumGatherPointFeature * sizeof(float))); //dev_pfe_gather_feature_
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pfe_buffers_[1]), kMaxNumPillars * sizeof(int)));  // voxel_count_list
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pfe_buffers_[2]), kMaxNumPoints * sizeof(int)));  // pid_to_dvid_map
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pfe_buffers_[3]), kMaxNumPillars * 2 * sizeof(int)));  // voxel_coors

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pfe_buffers_[4]), 64 * 400 * 400 * sizeof(float)));  // sparse_voxel_feat


    rpn_buffers_[0] = pfe_buffers_[4];
    GPU_CHECK(cudaMalloc(&rpn_buffers_[1],  kNumFeature * kAnchorSizes.size() * kAnchorRotations.size() * kNumClass * sizeof(float)));  //cls_score
    GPU_CHECK(cudaMalloc(&rpn_buffers_[2],  kNumFeature * kAnchorSizes.size() * kAnchorRotations.size() * kNumOutputBoxFeature * sizeof(float))); // bbox_pred
    GPU_CHECK(cudaMalloc(&rpn_buffers_[3],  kNumFeature * kAnchorSizes.size() * kAnchorRotations.size() * 2 * sizeof(float))); // dir_cls_pred

    // inputs, voxel_count_list, pid_to_dvid_map, voxel_coors
    // pfe_buffers_[0] = dev_pfe_gather_feature_;
    // preprocess_points_cuda_ptr_->SetDvInput(pfe_buffers_);

}


PointPillars::~PointPillars() {
    // for pillars 
    GPU_CHECK(cudaFree(dev_points_));

    // TODO: 优化Free方式
    GPU_CHECK(cudaFree(pfe_buffers_[0]));
    GPU_CHECK(cudaFree(pfe_buffers_[1]));
    GPU_CHECK(cudaFree(pfe_buffers_[2]));
    GPU_CHECK(cudaFree(pfe_buffers_[3]));
    GPU_CHECK(cudaFree(pfe_buffers_[4]));

    // GPU_CHECK(cudaFree(rpn_buffers_[0]));
    GPU_CHECK(cudaFree(rpn_buffers_[1]));
    GPU_CHECK(cudaFree(rpn_buffers_[2]));
    GPU_CHECK(cudaFree(rpn_buffers_[3]));

    pfe_context_->destroy();
    backbone_context_->destroy();
    pfe_engine_->destroy();
    backbone_engine_->destroy();
    // for post process
    // Destroy CUDA events
    cudaEventDestroy(preprocess_start_);
    cudaEventDestroy(preprocess_end_);
    cudaEventDestroy(pfe_start_);
    cudaEventDestroy(pfe_end_);
    cudaEventDestroy(scatter_start_);
    cudaEventDestroy(scatter_end_);
    cudaEventDestroy(backbone_start_);
    cudaEventDestroy(backbone_end_);
    cudaEventDestroy(postprocess_start_);
    cudaEventDestroy(postprocess_end_);
    cudaStreamDestroy(stream_);

}

void PointPillars::SetDeviceMemoryToZero() {
    GPU_CHECK(cudaMemsetAsync(pfe_buffers_[1], 0,  kMaxNumPillars * sizeof(int), stream_)); //voxel_count_list_
    GPU_CHECK(cudaMemsetAsync(pfe_buffers_[3], -1,  kMaxNumPillars * 2 * sizeof(int), stream_));
}

void PointPillars::InitTRT(const bool use_onnx) {
  if (use_onnx_) {
    // create a TensorRT model from the onnx model and load it into an engine
    OnnxToTRTModel(pfe_file_, &pfe_engine_);
    OnnxToTRTModel(backbone_file_, &backbone_engine_);
  }else {
    EngineToTRTModel(pfe_file_, &pfe_engine_);
    EngineToTRTModel(backbone_file_, &backbone_engine_);
  }
    if (pfe_engine_ == nullptr || backbone_engine_ == nullptr) {
        std::cerr << "Failed to load ONNX file.";
    }

    // create execution context from the engine
    pfe_context_ = pfe_engine_->createExecutionContext();
    backbone_context_ = backbone_engine_->createExecutionContext();
    if (pfe_context_ == nullptr || backbone_context_ == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context.";
    }
  
}

void PointPillars::OnnxToTRTModel(
    const std::string& model_file,  // name of the onnx model
    nvinfer1::ICudaEngine** engine_ptr) {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition* network =
        builder->createNetworkV2(explicit_batch);

    // parse onnx model
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 25);
    nvinfer1::ICudaEngine* engine =
        builder->buildEngineWithConfig(*network, *config);

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}


void PointPillars::EngineToTRTModel(
    const std::string &engine_file ,     
    nvinfer1::ICudaEngine** engine_ptr)  {
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream gieModelStream; 
    gieModelStream.seekg(0, gieModelStream.beg); 

    std::ifstream cache(engine_file); 
    gieModelStream << cache.rdbuf();
    cache.close(); 
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_); 

    if (runtime == nullptr) {
        std::string msg("failed to build runtime parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg(); 

    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize); 
    gieModelStream.read((char*)modelMem, modelSize);


    std::cout << " |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> "<< std::endl;
    std::cout << " | " << engine_file << " >" <<  std::endl;
    std::cout << " |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> "<< std::endl;
    std::cout << "             (\\__/) ||                 "<< std::endl;
    std::cout << "             (•ㅅ•) ||                 "<< std::endl;
    std::cout << "             / 　 づ                    "<< std::endl;
    
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL); 
    if (engine == nullptr) {
        std::string msg("failed to build engine parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    *engine_ptr = engine;

}

std::vector<BoundingBox> PointPillars::DoInference(const float* in_points_array, int in_num_points) 
{
    GPU_CHECK(cudaEventRecord(preprocess_start_, stream_));
    // [STEP 1] : load pointcloud
    // TODO: 初始化时malloc
    SetDeviceMemoryToZero();
    if (in_num_points > kMaxNumPoints){
        printf("[WARNING] Input points %d exceeds max limit %d, truncated to %d\n", 
            in_num_points, kMaxNumPoints, kMaxNumPoints); 
        in_num_points = kMaxNumPoints;
    }
    cudaMemcpyAsync(dev_points_, in_points_array, in_num_points * kNumPointFeature * sizeof(float),
        cudaMemcpyHostToDevice, stream_);
    // [STEP 2] : preprocess
    preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(dev_points_, in_num_points, pfe_buffers_, stream_);
    GPU_CHECK(cudaEventRecord(preprocess_end_, stream_));
    
    // [STEP 3] : pfe forward
    nvinfer1::Dims dims0;
    dims0.nbDims = 2;
    dims0.d[0] = in_num_points;
    dims0.d[1] = 10;
    
    nvinfer1::Dims dims1;
    dims1.nbDims = 1;
    dims1.d[0] = 40000;
    
    nvinfer1::Dims dims2;
    dims2.nbDims = 1;
    dims2.d[0] = in_num_points;
    
    nvinfer1::Dims dims3;
    dims3.nbDims = 2;
    dims3.d[0] = 40000;
    dims3.d[1] = 2;
    
    pfe_context_->setBindingDimensions(0, dims0);   // features
    pfe_context_->setBindingDimensions(1, dims1);   // num_voxels
    pfe_context_->setBindingDimensions(2, dims2);   // num_points
    pfe_context_->setBindingDimensions(3, dims3);   // coords

    GPU_CHECK(cudaEventRecord(pfe_start_, stream_));
    pfe_context_->enqueueV2(pfe_buffers_, stream_, nullptr);

    // 等待 PFE 完成，因为 scatter 需要其输出
    GPU_CHECK(cudaEventRecord(pfe_end_, stream_));

    // [STEP 4] : scatter pillar feature
    GPU_CHECK(cudaEventRecord(scatter_start_, stream_));
    // scatter_cuda_ptr_->DoScatterCuda(
    //     40000, reinterpret_cast<float*>(rpn_buffers_[1]), dev_scattered_feature_, stream_);
    // 等待 scatter 完成，因为 backbone 需要其输出
    GPU_CHECK(cudaEventRecord(scatter_end_, stream_));

    // [STEP 5] : backbone forward
    GPU_CHECK(cudaEventRecord(backbone_start_, stream_));
    backbone_context_->enqueueV2(rpn_buffers_, stream_, nullptr);
    // 等待 backbone 完成，因为 postprocess 需要其输出
    GPU_CHECK(cudaEventRecord(backbone_end_, stream_));

    // [STEP 6]: postprocess (multihead)
    GPU_CHECK(cudaEventRecord(postprocess_start_, stream_));
    postprocess_ptr_->DoPostprocess(
        reinterpret_cast<float*>(rpn_buffers_[1]), //cls_score
        reinterpret_cast<float*>(rpn_buffers_[2]),  // bbox_pred
        reinterpret_cast<float*>(rpn_buffers_[3]), // dir_cls_pred   
        stream_);
    // 等待 postprocess 完成，因为需要读取结果
    GPU_CHECK(cudaEventRecord(postprocess_end_, stream_));
    
    // 同步最后一个事件，确保所有操作都已完成，然后才能安全地计算时间
    GPU_CHECK(cudaEventSynchronize(postprocess_end_));

    // Calculate elapsed time using CUDA events
    float preprocess_time_ms = 0.0f, pfe_time_ms = 0.0f, scatter_time_ms = 0.0f;
    float backbone_time_ms = 0.0f, postprocess_time_ms = 0.0f, total_time_ms = 0.0f;
    
    GPU_CHECK(cudaEventElapsedTime(&preprocess_time_ms, preprocess_start_, preprocess_end_));
    GPU_CHECK(cudaEventElapsedTime(&pfe_time_ms, pfe_start_, pfe_end_));
    GPU_CHECK(cudaEventElapsedTime(&scatter_time_ms, scatter_start_, scatter_end_));
    GPU_CHECK(cudaEventElapsedTime(&backbone_time_ms, backbone_start_, backbone_end_));
    GPU_CHECK(cudaEventElapsedTime(&postprocess_time_ms, postprocess_start_, postprocess_end_));
    GPU_CHECK(cudaEventElapsedTime(&total_time_ms, preprocess_start_, postprocess_end_));

    std::cout << "------------------------------------" << std::endl;
    std::cout << setiosflags(ios::left)  << setw(14) << "Module" << setw(12)  << "Time"  << resetiosflags(ios::left) << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::string Modules[] = {"Preprocess" , "Pfe" , "Scatter" , "Backbone" , "Postprocess" , "Summary"};
    double Times[] = {preprocess_time_ms, pfe_time_ms, scatter_time_ms, backbone_time_ms, postprocess_time_ms, total_time_ms}; 

    for (int i =0 ; i < 6 ; ++i) {
        std::cout << setiosflags(ios::left) << setw(14) << Modules[i]  << setw(8)  << Times[i] << " ms" << resetiosflags(ios::left) << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
    return this->postprocess_ptr_->bndBoxVec();
}