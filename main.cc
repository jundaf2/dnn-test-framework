#include "linear_layer.h"

#include <iostream>
#include <fstream>
#include "torch/torch.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "nn_test.h"
#include "cublas_v2.h"

using namespace std;

struct test_Linear : public nn_test::nnTest, torch::nn::Module {

  test_Linear(int batch, int in_features, int out_features){
    this->batch = batch;
    this->in_features  = in_features;
    this->out_features  = out_features;
    // Construct and register  submodules in a Torch manner.
    this->linear = register_module("linear", torch::nn::Linear(in_features, out_features));
  }

public:
  void init_data() override {
    size_t weight_len = in_features*out_features;
    size_t bias_len = out_features;
    size_t in_data_len = batch*in_features;
    size_t out_data_len = batch*out_features;

    unsigned int seed = 2023;
    float rand_range = 2;
    this->set_random_seed(seed);
    this->set_print_el_num(64);
    // weight and bias 
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,weight_len).data(), weight_len, "linear_weight");
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,bias_len).data(), bias_len, "linear_bias");

    // input
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,in_data_len).data(), in_data_len, "linear_in");

    // target
    this->set_input_vec(this->gen_rand_input(-rand_range,rand_range,out_data_len).data(), out_data_len, "target");
  }

  
  void run_my_dnn() override{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // host ptr
    float *linear_in_grad, *linear_weight_grad, *linear_bias_grad, *output;
    linear_in_grad = (float*) malloc(batch*in_features*sizeof(float));
    linear_weight_grad = (float*) malloc(out_features*in_features*sizeof(float));
    linear_bias_grad = (float*) malloc(out_features*sizeof(float));
    output = (float*) malloc(batch*out_features*sizeof(float));

    // deivce ptr
    float *d_linear_in, *d_linear_weight, *d_linear_bias, *d_output, *d_target;
    float *d_linear_in_grad, *d_linear_weight_grad, *d_linear_bias_grad, *d_output_grad;
    float *d_loss;
    
    cudaStat = cudaMalloc((void**)&d_linear_in, batch*in_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_linear_weight, out_features*in_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_linear_bias, out_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_target, batch*out_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_output, batch*out_features*sizeof(float));

    cudaStat = cudaMalloc((void**)&d_linear_in_grad, batch*in_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_linear_weight_grad, out_features*in_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_linear_bias_grad, out_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_output_grad, batch*out_features*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_loss, sizeof(float));

    // copy h2d
    cudaStat = cudaMemcpy(d_linear_in,this->get_input_vec("linear_in").data(),batch*in_features*sizeof(float),cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(d_linear_weight,this->get_input_vec("linear_weight").data(),out_features*in_features*sizeof(float),cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(d_linear_bias,this->get_input_vec("linear_bias").data(),out_features*sizeof(float),cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(d_target,this->get_input_vec("target").data(),batch*out_features*sizeof(float),cudaMemcpyHostToDevice);

    float alpha = 1.;
    float beta = 0.;
    // forward
    stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch, in_features, (const void *)&alpha,
                   (const void *)d_linear_weight, CUDA_R_32F, in_features, (const void *)d_linear_in, CUDA_R_32F, in_features,
                   (const void *)&beta, d_output, CUDA_R_32F, out_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    
    // add bias
    launch_linear_bias_fwd_kernel(d_output, d_linear_bias, out_features, batch);

    // mse loss
    launch_mse_loss_kernel(d_output,d_target,d_loss,d_output_grad,batch*out_features);

    // backward
    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, batch, (const void *)&alpha,
                   (const void *)d_linear_in, CUDA_R_32F, in_features, (const void *)d_output_grad, CUDA_R_32F, out_features,
                   (const void *)&beta, d_linear_weight_grad, CUDA_R_32F, in_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, batch, out_features, (const void *)&alpha,
                   (const void *)d_linear_weight, CUDA_R_32F, in_features, (const void *)d_output_grad, CUDA_R_32F, out_features,
                   (const void *)&beta, d_linear_in_grad, CUDA_R_32F, in_features, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    launch_linear_bias_bwd_kernel(d_output_grad, d_linear_bias_grad, out_features, batch);

    // copy d2h
    cudaStat = cudaMemcpy(linear_in_grad,d_linear_in_grad,batch*in_features*sizeof(float),cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(linear_weight_grad,d_linear_weight_grad,out_features*in_features*sizeof(float),cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(linear_bias_grad,d_linear_bias_grad,out_features*sizeof(float),cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(output,d_output,batch*out_features*sizeof(float),cudaMemcpyDeviceToHost);

    cudaStat = cudaFree(d_linear_in);
    cudaStat = cudaFree(d_linear_weight);
    cudaStat = cudaFree(d_linear_bias);
    cudaStat = cudaFree(d_target);
    cudaStat = cudaFree(d_output);

    cudaStat = cudaFree(d_output_grad);
    cudaStat = cudaFree(d_linear_in_grad);
    cudaStat = cudaFree(d_linear_weight_grad);
    cudaStat = cudaFree(d_linear_bias_grad);

    cublasDestroy(handle);


    // Register the data to be test
    this->register_raw_test_data(output, batch*out_features, "output");
    this->register_raw_test_data(linear_in_grad, batch*in_features, "linear_in_grad");
    this->register_raw_test_data(linear_weight_grad, in_features*out_features, "linear_weight_grad");
    this->register_raw_test_data(linear_bias_grad, out_features, "linear_bias_grad");

    free(output);
    free(linear_in_grad);
    free(linear_weight_grad);
    free(linear_bias_grad);

  }

  void run_torch_dnn() override{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(true);
    // Init Input Data Tensor
    torch::Tensor linear_in = torch::empty({batch,in_features}, options); 
    this->get_input_ten(linear_in, "linear_in", options);

    // Init Target Data Tensor
    torch::Tensor target = torch::empty({batch,out_features});
    this->get_input_ten(target, "target", options);

    // Init Input Weight and Bias Tensor
    this->get_input_ten(this->linear->weight, "linear_weight", options);  
    this->get_input_ten(this->linear->bias, "linear_bias", options);

    torch::Tensor linear_out = this->forward(linear_in);

    torch::Tensor loss = torch::mse_loss(target, linear_out);
    loss.backward();

    // Register the data to be compared with
    this->register_torch_test_data(linear_out, "output");
    this->register_torch_test_data(linear_in.grad(), "linear_in_grad");
    this->register_torch_test_data(this->linear->weight.grad(), "linear_weight_grad");
    this->register_torch_test_data(this->linear->bias.grad(), "linear_bias_grad");
  }

  // Implement the Linear's algorithm in a Torch manner.
  torch::Tensor forward(torch::Tensor input) {
    return this->linear->forward(input);
  }

private:
  torch::nn::Linear linear{nullptr};
  unsigned batch, in_features, out_features;
};

int eval_linear(unsigned batch, unsigned in_features,unsigned out_features){
  test_Linear test_linear(batch,in_features,out_features);
  test_linear.init_data();
  test_linear.run_my_dnn();
  test_linear.run_torch_dnn();
  test_linear.verify();
}

TEST_CASE("Linear", "[Linear Layer]") {
  SECTION("[4,50,100]") {
    eval_linear(4,50,100);
  }
}