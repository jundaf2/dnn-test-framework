void launch_linear_bias_fwd_kernel(float* x, float* bias, int out_features, int batch);

void launch_linear_bias_bwd_kernel(const float* dx, float* d_bias, int out_features, int batch);

void launch_mse_loss_kernel(const float* output, const float* target, float* loss, float* d_loss, int num_elem);
