__global__ void voxel_mean_kernel(
    const float* point_feats,
    const int* voxel_count_list,
    const int* pid_to_dvid_map,
    int num_points,
    int num_feats,
    float* reduced_feat)
  {
    int p_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_id >= num_points) return;
    int dvid = pid_to_dvid_map[p_id];
    if (dvid == -1) return;
    int cnt = voxel_count_list[dvid];
    float inv = 1.0f / cnt;
    const float* point_feat_offset = point_feats + p_id * num_feats;
    float* reduced_feat_offset = reduced_feat + dvid * num_feats;
    #pragma unroll
    for (int c = 0; c < num_feats; ++c) {
        atomicAdd(&reduced_feat_offset[c], point_feat_offset[c] * inv);
    }
  }

  // __global__ void voxel_mean_kernel2(
  //   const float* point_feats,
  //   const int* voxel_count_list,
  //   const int* pid_to_dvid_map,
  //   int num_points,
  //   int num_feats,
  //   float* reduced_feat)
  // {
  //   int p_id = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (p_id >= num_points) return;
  //   int dvid = pid_to_dvid_map[p_id];
  //   if (dvid == -1) return;
  //   int cnt = voxel_count_list[dvid];
  //   float inv = 1.0f / cnt;
  //   const float* point_feat_offset = point_feats + p_id * num_feats;
  //   float* reduced_feat_offset = reduced_feat + dvid * num_feats;
  //   #pragma unroll
  //   for (int c = 0; c < num_feats; ++c) {
  //       atomicAdd(&reduced_feat_offset[c], point_feat_offset[c] * inv);
  //   }

  //   #ifndef NDEBUG
  //   if (p_id == 40000)
  //   { 
  //     printf("p_id = %d, point_feat_offset =", p_id);
  //     for (int c = 0; c < num_feats; ++c) {
  //       printf("%.3f  ", point_feat_offset[c]);
  //     }
  //     printf("\n");

  //     for (int c = 0; c < 100000; ++c) {
  //       inv += 1.0f / cnt;
  //     }
  //     printf("reduced_feat_offset %.3f  %.3f  %.3f  %.3f \n", 
  //       reduced_feat_offset[0], reduced_feat_offset[1], reduced_feat_offset[2], reduced_feat_offset[3]);
  //   }
  //   #endif


  //   // #ifndef NDEBUG
  //   // if (dvid == 0)
  //   // { 
  //   //   printf("num_points = %d \n", num_points), 

  //   //   printf("p_id = %d, point_feat_offset =  %.3f  %.3f  %.3f  %.3f \n", 
  //   //     p_id, point_feat_offset[0], point_feat_offset[1], point_feat_offset[2], point_feat_offset[3]);
  //   //   for (int c = 0; c < 100000; ++c) {
  //   //     inv += 1.0f / cnt;
  //   //   }
  //   //   printf("reduced_feat_offset %.3f  %.3f  %.3f  %.3f \n", 
  //   //     reduced_feat_offset[0], reduced_feat_offset[1], reduced_feat_offset[2], reduced_feat_offset[3]);
  //   // }
  //   // #endif

  // }

  __global__ void get_feat_per_point_kernel(
    const float* reduced_feat,
    const int* pid_to_dvid_map,
    int num_points,
    int num_feats,
    float* feat_per_point)
  {
    int p_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_id >= num_points) return;
    int dvid = pid_to_dvid_map[p_id];
    if (dvid == -1) return;
    float* feat_per_point_offset = feat_per_point + p_id * num_feats;
    const float* reduced_feat_offset = reduced_feat + dvid * num_feats;
    #pragma unroll
    for (int c = 0; c < num_feats; ++c) {
        feat_per_point_offset[c] = reduced_feat_offset[c];
    }  
  }
