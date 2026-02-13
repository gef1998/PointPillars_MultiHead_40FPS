/**
 * @file get_feat_per_point_plugin.cc
 * @brief TensorRT GetFeatPerPoint 插件的框架实现(不含具体计算逻辑)。
 */

#include "get_feat_per_point_plugin.h"
#include "NvInferPlugin.h"
#include "common.h"

#include <cassert>
#include <cstring>

#include <cuda_runtime.h>

// CUDA kernel 前向声明, 实现见 `get_feat_per_point.cu`
__global__ void voxel_mean_kernel(
    const float* point_feats,
    const int* voxel_count_list,
    const int* pid_to_dvid_map,
    int num_points,
    int num_feats,
    float* reduced_feat);

__global__ void get_feat_per_point_kernel(
    const float* reduced_feat,
    const int* pid_to_dvid_map,
    int num_points,
    int num_feats,
    float* feat_per_point);

namespace {

static const char* GET_FEAT_PER_POINT_PLUGIN_NAME{"GetFeatPerPoint"};
static const char* GET_FEAT_PER_POINT_PLUGIN_VERSION{"1"};

}  // namespace

namespace nvinfer1 {

// 静态成员定义
PluginFieldCollection GetFeatPerPointPluginCreator::mFC_{};
std::vector<PluginField> GetFeatPerPointPluginCreator::mPluginAttributes_{};

// ============================== GetFeatPerPointPlugin ==============================

GetFeatPerPointPlugin::GetFeatPerPointPlugin(const std::string& name)
    : mLayerName_(name) {}

GetFeatPerPointPlugin::GetFeatPerPointPlugin(const std::string& name,
                                             const void* /*data*/,
                                             size_t /*length*/)
    : mLayerName_(name) {}

const char* GetFeatPerPointPlugin::getPluginType() const noexcept {
  return GET_FEAT_PER_POINT_PLUGIN_NAME;
}

const char* GetFeatPerPointPlugin::getPluginVersion() const noexcept {
  return GET_FEAT_PER_POINT_PLUGIN_VERSION;
}

int GetFeatPerPointPlugin::getNbOutputs() const noexcept { return 1; }

int GetFeatPerPointPlugin::initialize() noexcept { return 0; }

void GetFeatPerPointPlugin::terminate() noexcept {}

size_t GetFeatPerPointPlugin::getSerializationSize() const noexcept {
  // 当前没有需要序列化的额外参数
  return 0;
}

void GetFeatPerPointPlugin::serialize(void* /*buffer*/) const noexcept {
  // 当前没有需要序列化的额外参数
}

void GetFeatPerPointPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt* GetFeatPerPointPlugin::clone() const noexcept {
  try {
    auto* plugin = new GetFeatPerPointPlugin(mLayerName_);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

void GetFeatPerPointPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mNamespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GetFeatPerPointPlugin::getPluginNamespace() const noexcept {
  return mNamespace_.c_str();
}

DataType GetFeatPerPointPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept {
  assert(nbInputs == 3);
  assert(index == 0);
  // 默认输出与 point_feats 的数据类型保持一致
  return inputTypes[0];
}

DimsExprs GetFeatPerPointPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& /*exprBuilder*/) noexcept {
  assert(nbInputs == 3);
  assert(outputIndex == 0);
  // 输出维度与 point_feats (输入0) 保持一致
  return inputs[0];
}

bool GetFeatPerPointPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) noexcept {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  const PluginTensorDesc& desc = inOut[pos];
  // 统一只支持 LINEAR
  if (desc.format != nvinfer1::TensorFormat::kLINEAR)
      return false;

  switch (pos)
  {
  case 0:
      // point_feats: FP32 / FP16 都支持
      return desc.type == nvinfer1::DataType::kFLOAT;
      //  || desc.type == nvinfer1::DataType::kHALF;

  case 1:
      // voxel_count_list: INT32
      return desc.type == nvinfer1::DataType::kINT32;

  case 2:
      // pid_to_dvid_map: INT32
      return desc.type == nvinfer1::DataType::kINT32;

  case 3:
      // 输出 sparse_voxel_feat
      return desc.type == inOut[0].type;

  default:
      return false;
  }
}


void GetFeatPerPointPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  // 这里可以做输入/输出 shape 检查, 当前仅做最基本的断言
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  (void)in;
  (void)out;
}

size_t GetFeatPerPointPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* /*outputs*/,
    int /*nbOutputs*/) const noexcept {
  // reduced_feat: [num_voxels, num_feats] float
  // inputs[0]: point_feats [num_points, num_feats]
  // inputs[1]: voxel_count_list [num_voxels]
  // inputs[2]: pid_to_dvid_map [num_points]
  assert(nbInputs == 3);
  const int num_voxels = inputs[1].dims.d[0];
  const int num_feats = inputs[0].dims.d[1];
  return static_cast<size_t>(num_voxels) * static_cast<size_t>(num_feats) * sizeof(float);
}

int GetFeatPerPointPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                   const PluginTensorDesc* outputDesc,
                                   const void* const* inputs,
                                   void* const* outputs,
                                   void* workspace,
                                   cudaStream_t stream) noexcept {
  (void)outputDesc;

  // inputs:
  //  0: point_feats [num_points, num_feats] float
  //  1: voxel_count_list [num_voxels] int32
  //  2: pid_to_dvid_map [num_points] int32
  const Dims& point_feats_dims = inputDesc[0].dims;
  const int num_points = point_feats_dims.d[0];
  const int num_feats = point_feats_dims.d[1];
  const int num_voxels = inputDesc[1].dims.d[0];

  const auto* point_feats = static_cast<const float*>(inputs[0]);
  const auto* voxel_count_list = static_cast<const int*>(inputs[1]);
  const auto* pid_to_dvid_map = static_cast<const int*>(inputs[2]);
  auto* feat_per_point = static_cast<float*>(outputs[0]);

  // workspace 用作 reduced_feat: [num_voxels, num_feats]
  auto* reduced_feat = static_cast<float*>(workspace);
  const size_t reduced_bytes =
      static_cast<size_t>(num_voxels) * static_cast<size_t>(num_feats) * sizeof(float);

  // 初始化 reduced_feat 为 0，保证后续 atomicAdd 累加正确
  cudaMemsetAsync(reduced_feat, 0, reduced_bytes, stream);

  const int threads = 64;
  const int num_point_block = DIVUP(num_points, threads);
  voxel_mean_kernel<<<num_point_block, threads, 0, stream>>>(
      point_feats, voxel_count_list, pid_to_dvid_map, num_points, num_feats, reduced_feat);
  get_feat_per_point_kernel<<<num_point_block, threads, 0, stream>>>(
      reduced_feat, pid_to_dvid_map, num_points, num_feats, feat_per_point);
  
  return 0;
}

// ============================== GetFeatPerPointPluginCreator ==============================

GetFeatPerPointPluginCreator::GetFeatPerPointPluginCreator() {
  mPluginAttributes_.clear();
  mFC_.nbFields = 0;
  mFC_.fields = nullptr;
}

const char* GetFeatPerPointPluginCreator::getPluginName() const noexcept {
  return GET_FEAT_PER_POINT_PLUGIN_NAME;
}

const char* GetFeatPerPointPluginCreator::getPluginVersion() const noexcept {
  return GET_FEAT_PER_POINT_PLUGIN_VERSION;
}

const PluginFieldCollection* GetFeatPerPointPluginCreator::getFieldNames() noexcept {
  return &mFC_;
}

IPluginV2DynamicExt* GetFeatPerPointPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* /*fc*/) noexcept {
  try {
    auto* plugin = new GetFeatPerPointPlugin(name);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

IPluginV2DynamicExt* GetFeatPerPointPluginCreator::deserializePlugin(
    const char* name, const void* serialData,
    size_t serialLength) noexcept {
  try {
    auto* plugin = new GetFeatPerPointPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

void GetFeatPerPointPluginCreator::setPluginNamespace(
    const char* pluginNamespace) noexcept {
  mNamespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GetFeatPerPointPluginCreator::getPluginNamespace() const noexcept {
  return mNamespace_.c_str();
}

// 在 TensorRT 插件注册表中注册该插件
REGISTER_TENSORRT_PLUGIN(GetFeatPerPointPluginCreator);

}  // namespace nvinfer1


