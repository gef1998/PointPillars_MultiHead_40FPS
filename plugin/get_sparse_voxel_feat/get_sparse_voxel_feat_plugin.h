/**
 * @file get_sparse_voxel_feat_plugin.h
 * @brief TensorRT plugin skeleton for GetSparseVoxelFeat operation.
 *
 * 输入:
 *  - point_feats
 *  - voxel_count_list
 *  - pid_to_dvid_map
 *  - voxel_coors
 * 输出:
 *  - sparse_voxel_feat
 *
 * 本文件只包含插件框架与接口定义, 具体实现逻辑在 .cc 中补充。
 */

#pragma once

#include "NvInfer.h"

#include <string>
#include <vector>

namespace nvinfer1 {

class GetSparseVoxelFeatPlugin final : public IPluginV2DynamicExt {
 public:
  // 用于从网络定义阶段创建插件
  explicit GetSparseVoxelFeatPlugin(const std::string& name);

  // 用于反序列化创建插件
  GetSparseVoxelFeatPlugin(const std::string& name, const void* data, size_t length);

  // ---------- IPluginV2 接口 ----------
  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  int getNbOutputs() const noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void* buffer) const noexcept override;

  void destroy() noexcept override;

  IPluginV2DynamicExt* clone() const noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

  // ---------- IPluginV2Ext 接口 ----------
  DataType getOutputDataType(int index, const DataType* inputTypes,
                             int nbInputs) const noexcept override;

  // ---------- IPluginV2DynamicExt 接口 ----------
  DimsExprs getOutputDimensions(
      int outputIndex, const DimsExprs* inputs, int nbInputs,
      IExprBuilder& exprBuilder) noexcept override;

  bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) noexcept override;

  void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                       const DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;

  size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                          const PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;

  int enqueue(const PluginTensorDesc* inputDesc,
              const PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

 private:
  std::string mLayerName_;
  std::string mNamespace_;
};

class GetSparseVoxelFeatPluginCreator final : public IPluginCreator {
 public:
  GetSparseVoxelFeatPluginCreator();

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const PluginFieldCollection* getFieldNames() noexcept override;

  IPluginV2DynamicExt* createPlugin(const char* name,
                                    const PluginFieldCollection* fc) noexcept override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

 private:
  static PluginFieldCollection mFC_;
  static std::vector<PluginField> mPluginAttributes_;

  std::string mNamespace_;
};

}  // namespace nvinfer1


