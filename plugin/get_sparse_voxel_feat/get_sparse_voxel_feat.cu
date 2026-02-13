__global__ void get_sparse_voxel_feat_kernel(
    const float* reduced_feat,
    const int* voxel_coors,
    int num_feats,
    float* sparse_voxel_feat)
{
  int v_id = blockIdx.x;
  int i_feature = threadIdx.x;

  int y_ind = voxel_coors[v_id * 2 + 0];
  if (y_ind == -1){
    return;
  }
  int x_ind = voxel_coors[v_id * 2 + 1];
  // TODO: 400can
  sparse_voxel_feat[i_feature * 400 * 400 + y_ind * 400 + x_ind] = reduced_feat[v_id * num_feats + i_feature]; 

}
