#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"


/* forward bias kernel
@ x is a 2d data with shape [out_features,batch]
@ bias is a 1d data with shape [out_features]
*/
__global__ void linear_bias_fwd_kernel(float* x, float* bias, int out_features, int batch){
  int out_features_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(out_features_idx<out_features){
    int this_warp_bias = bias[out_features_idx];
    for(int batch_idx=threadIdx.y; batch_idx<batch; batch_idx+=blockDim.y){
      x[batch_idx*out_features+out_features_idx] += this_warp_bias;
    }
  }
}

// backward bias kernel
__global__ void linear_bias_bwd_kernel(float* dx, float* d_bias, int out_features, int batch){

}

// mse loss kernel
__global__ void mse_loss_kernel(float* output, float* target){

}

void launch_linear_bias_kernel(float* x, float* bias, int out_features, int batch){
  dim3 grid_dim((out_features - 1) / 32 + 1);
  dim3 block_dim(32, 32);
  linear_bias_kernel<<<grid_dim, block_dim>>>(x, bias, out_features, batch);
}