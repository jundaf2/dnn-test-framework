#include <cuda.h>
#include <cuda_runtime.h>
#include "linear_layer.h"

/* forward bias kernel
@ x is a 2d data with shape [batch, out_features]
@ bias is a 1d data with shape [out_features]
*/
__global__ void linear_bias_fwd_kernel(float* x, float* bias, int out_features, int batch){
  int batch_idx = blockIdx.x;

  for(int out_features_idx=threadIdx.x; out_features_idx<out_features; out_features_idx+=blockDim.x){
    float this_bias = bias[out_features_idx];
    x[batch_idx*out_features+out_features_idx] += this_bias;
  }
}

/* backward bias kernel
@ dx is 2d data with shape [batch, out_features]
@ d_bias is 1d data with shape [out_features]
*/
__global__ void linear_bias_bwd_kernel(const float* dx, float* d_bias, int out_features, int batch){
  int out_features_idx = blockIdx.x*blockDim.x + threadIdx.x;
  float thread_sum;

  if(out_features_idx<out_features){
    for(int batch_idx=0; batch_idx<batch; batch_idx++){
      thread_sum += dx[batch_idx*out_features+out_features_idx];
    }
    d_bias[out_features_idx] = thread_sum;
  }
}

/* mse loss kernel
@ target, output, d_loss are is 2d data [batch,out_features]
@ loss is a scalar
*/
__global__ void mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx==0) *loss=0;

  if(idx<num_elem)
  {
    float err = output[idx] - target[idx];
    float err2 = err * err;
    float mean_square_error = err2/num_elem;
    atomicAdd(loss, mean_square_error); // poor performance
    d_loss[idx] = 2 * err * (1.0f/num_elem);
  }
}



void launch_linear_bias_fwd_kernel(float* x, float* bias, int out_features, int batch){
  dim3 blocks(batch);
  dim3 threads(512);
  linear_bias_fwd_kernel<<<blocks, threads>>>(x, bias, out_features, batch);
  cudaDeviceSynchronize();
}

void launch_linear_bias_bwd_kernel(const float* dx, float* d_bias, int out_features, int batch){
  dim3 blocks((out_features - 1) / 512 + 1);
  dim3 threads(512);
  linear_bias_bwd_kernel<<<blocks, threads>>>(dx, d_bias, out_features, batch);
  cudaDeviceSynchronize();
}

void launch_mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem){
  dim3 blocks((num_elem - 1) / 512 + 1);
  dim3 threads(512);
  mse_loss_kernel<<<blocks, threads>>>(output, target, loss, d_loss, num_elem);
  cudaDeviceSynchronize();
}
