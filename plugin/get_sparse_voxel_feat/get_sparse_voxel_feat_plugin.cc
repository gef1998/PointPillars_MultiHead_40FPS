/**
 * @file get_sparse_voxel_feat_plugin.cc
 * @brief TensorRT GetSparseVoxelFeat 插件的框架实现
 */

#include "get_sparse_voxel_feat_plugin.h"
#include "NvInferPlugin.h"
#include "common.h"

#include <cassert>
#include <cstring>

#include <cuda_runtime.h>

// CUDA kernel 前向声明, 实现见 `get_sparse_voxel_feat.cu`
__global__ void voxel_mean_kernel(
    const float* point_feats,
    const int* voxel_count_list,
    const int* pid_to_dvid_map,
    int num_points,
    int num_feats,
    float* reduced_feat);

__global__ void get_sparse_voxel_feat_kernel(
    const float* reduced_feat,
    const int* voxel_coors,
    int num_feats,
    float* sparse_voxel_feat);

namespace {

static const char* GET_SPARSE_VOXEL_FEAT_PLUGIN_NAME{"GetSparseVoxelFeat"};
static const char* GET_SPARSE_VOXEL_FEAT_PLUGIN_VERSION{"1"};

}  // namespace

namespace nvinfer1 {

// 静态成员定义
PluginFieldCollection GetSparseVoxelFeatPluginCreator::mFC_{};
std::vector<PluginField> GetSparseVoxelFeatPluginCreator::mPluginAttributes_{};

// ============================== GetSparseVoxelFeatPlugin ==============================

GetSparseVoxelFeatPlugin::GetSparseVoxelFeatPlugin(const std::string& name)
    : mLayerName_(name) {}

GetSparseVoxelFeatPlugin::GetSparseVoxelFeatPlugin(const std::string& name,
                                             const void* /*data*/,
                                             size_t /*length*/)
    : mLayerName_(name) {}

const char* GetSparseVoxelFeatPlugin::getPluginType() const noexcept {
  return GET_SPARSE_VOXEL_FEAT_PLUGIN_NAME;
}

const char* GetSparseVoxelFeatPlugin::getPluginVersion() const noexcept {
  return GET_SPARSE_VOXEL_FEAT_PLUGIN_VERSION;
}

int GetSparseVoxelFeatPlugin::getNbOutputs() const noexcept { return 1; }

int GetSparseVoxelFeatPlugin::initialize() noexcept { return 0; }

void GetSparseVoxelFeatPlugin::terminate() noexcept {}

size_t GetSparseVoxelFeatPlugin::getSerializationSize() const noexcept {
  // 当前没有需要序列化的额外参数
  return 0;
}

void GetSparseVoxelFeatPlugin::serialize(void* /*buffer*/) const noexcept {
  // 当前没有需要序列化的额外参数
}

void GetSparseVoxelFeatPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt* GetSparseVoxelFeatPlugin::clone() const noexcept {
  try {
    auto* plugin = new GetSparseVoxelFeatPlugin(mLayerName_);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

void GetSparseVoxelFeatPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mNamespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GetSparseVoxelFeatPlugin::getPluginNamespace() const noexcept {
  return mNamespace_.c_str();
}

DataType GetSparseVoxelFeatPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept {
  assert(nbInputs == 4);
  assert(index == 0);
  // 默认输出与 point_feats 的数据类型保持一致
  return inputTypes[0];
}

DimsExprs GetSparseVoxelFeatPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept {
  assert(nbInputs == 4);
  assert(outputIndex == 0);

  DimsExprs out;
  out.nbDims = 4;

  // N
  out.d[0] = exprBuilder.constant(1);

  // C
  out.d[1] = exprBuilder.constant(64);

  // H
  out.d[2] = exprBuilder.constant(400);

  // W
  out.d[3] = exprBuilder.constant(400);
  // 输出维度用onnx的attribute
  return out;
}

bool GetSparseVoxelFeatPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) noexcept {
  assert(nbInputs == 4);
  assert(nbOutputs == 1);
  const PluginTensorDesc& desc = inOut[pos];


    // 统一只支持 LINEAR
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    switch (pos)
    {
    case 0:
        // point_feats: FP32 / FP16 都支持
        return desc.type == nvinfer1::DataType::kFLOAT ||
               desc.type == nvinfer1::DataType::kHALF;

    case 1:
        // voxel_count_list: INT32
        return desc.type == nvinfer1::DataType::kINT32;

    case 2:
        // pid_to_dvid_map: INT32
        return desc.type == nvinfer1::DataType::kINT32;

    case 3:
        // voxel_coors: INT32
        return desc.type == nvinfer1::DataType::kINT32;

    case 4:
        // 输出 sparse_voxel_feat
        // ⚠️ 必须和 input0 的 dtype 一致
        return desc.type == inOut[0].type;

    default:
        return false;
    }
}

void GetSparseVoxelFeatPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  // 这里可以做输入/输出 shape 检查, 当前仅做最基本的断言
  assert(nbInputs == 4);
  assert(nbOutputs == 1);
  (void)in;
  (void)out;
}

size_t GetSparseVoxelFeatPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* /*outputs*/,
    int /*nbOutputs*/) const noexcept {
  // reduced_feat: [num_voxels, num_feats] float
  // inputs[0]: point_feats [num_points, num_feats]
  // inputs[1]: voxel_count_list [num_voxels]
  // inputs[2]: pid_to_dvid_map [num_points]
  // inputs[3]: voxel_coors [num_voxels, 2]
  assert(nbInputs == 4);
  const int num_voxels = inputs[1].dims.d[0];
  const int num_feats = inputs[0].dims.d[1];
  return static_cast<size_t>(num_voxels) * static_cast<size_t>(num_feats) * sizeof(float);
}

int GetSparseVoxelFeatPlugin::enqueue(const PluginTensorDesc* inputDesc,
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
  //  3: voxel_coors [num_voxels, 2] int32
  const Dims& point_feats_dims = inputDesc[0].dims;
  const int num_points = point_feats_dims.d[0];
  const int num_feats = point_feats_dims.d[1];
  const int num_voxels = inputDesc[1].dims.d[0];

  const auto* point_feats = static_cast<const float*>(inputs[0]);
  const auto* voxel_count_list = static_cast<const int*>(inputs[1]);
  const auto* pid_to_dvid_map = static_cast<const int*>(inputs[2]);
  const auto* voxel_coors = static_cast<const int*>(inputs[3]);
  auto* sparse_voxel_feat = static_cast<float*>(outputs[0]);

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

  get_sparse_voxel_feat_kernel<<<40000, num_feats, 0, stream>>>(reduced_feat, voxel_coors, num_feats, sparse_voxel_feat);

  return 0;
}

// ============================== GetSparseVoxelFeatPluginCreator ==============================

GetSparseVoxelFeatPluginCreator::GetSparseVoxelFeatPluginCreator() {
  mPluginAttributes_.clear();
  mFC_.nbFields = 0;
  mFC_.fields = nullptr;
}

const char* GetSparseVoxelFeatPluginCreator::getPluginName() const noexcept {
  return GET_SPARSE_VOXEL_FEAT_PLUGIN_NAME;
}

const char* GetSparseVoxelFeatPluginCreator::getPluginVersion() const noexcept {
  return GET_SPARSE_VOXEL_FEAT_PLUGIN_VERSION;
}

const PluginFieldCollection* GetSparseVoxelFeatPluginCreator::getFieldNames() noexcept {
  return &mFC_;
}

IPluginV2DynamicExt* GetSparseVoxelFeatPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* /*fc*/) noexcept {
  try {
    auto* plugin = new GetSparseVoxelFeatPlugin(name);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

IPluginV2DynamicExt* GetSparseVoxelFeatPluginCreator::deserializePlugin(
    const char* name, const void* serialData,
    size_t serialLength) noexcept {
  try {
    auto* plugin = new GetSparseVoxelFeatPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
  } catch (...) {
    return nullptr;
  }
}

void GetSparseVoxelFeatPluginCreator::setPluginNamespace(
    const char* pluginNamespace) noexcept {
  mNamespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GetSparseVoxelFeatPluginCreator::getPluginNamespace() const noexcept {
  return mNamespace_.c_str();
}

// 在 TensorRT 插件注册表中注册该插件
REGISTER_TENSORRT_PLUGIN(GetSparseVoxelFeatPluginCreator);

}  // namespace nvinfer1


