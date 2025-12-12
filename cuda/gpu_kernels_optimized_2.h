#ifndef GPU_KERNELS_OPTIMIZED_2_H
#define GPU_KERNELS_OPTIMIZED_2_H

// Sử dụng fused Conv2D + ReLU + Bias
void launch_conv2d_forward_relu_fused(const float* d_input, float* d_output,
                                      const float* d_weights, const float* d_bias,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding, cudaStream_t stream);

void launch_conv2d_forward(const float* d_input, float* d_output,
                           const float* d_weights, const float* d_bias,
                           int batch, int in_h, int in_w, int in_c,
                           int out_c, int kernel_size, int stride, int padding, cudaStream_t stream);

// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ như poolsize =2
void launch_maxpool2d_optimized_forward(const float* d_input, float* d_output, float* indices,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride, cudaStream_t stream);
                                        
void launch_upsample2d_forward(const float* d_input, float* d_output,
                               int batch, int in_h, int in_w, int c,
                               int scale_factor, cudaStream_t stream);

// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ + fused
void launch_conv2d_relu_backward_fused(const float* d_grad_output, 
                                       const float* d_input,
                                       const float* d_weights, 
                                       const float* d_conv_output,
                                       float* d_grad_input,
                                       float* d_grad_weights, 
                                       float* d_grad_bias,
                                       int batch, int in_h, int in_w, int in_c,
                                       int out_c, int kernel_size, int stride, int padding, cudaStream_t stream);
                                       
void launch_conv2d_backward(const float* d_grad_output, const float* d_input,
                            const float* d_weights, float* d_grad_input,
                            float* d_grad_weights, float* d_grad_bias,
                            int batch, int in_h, int in_w, int in_c,
                            int out_c, int kernel_size, int stride, int padding, cudaStream_t stream);

void launch_maxpool2d_backward(const float* d_grad_output, float* d_input, const float* indices,
                               const float* d_output, float* d_grad_input,
                               int batch, int h, int w, int c,
                               int pool_size, int stride, cudaStream_t stream);

void launch_upsample2d_backward(const float* d_grad_output, float* d_grad_input,
                                int batch, int in_h, int in_w, int c,
                                int scale_factor, cudaStream_t stream);

void launch_mse_loss(const float* d_input, const float* d_output,
                     float* d_loss, int size, cudaStream_t stream);

void launch_zero_grad(float* d_grad, int size, cudaStream_t stream);

void launch_sgd_update(float* d_weights, const float* d_grad,
                       float learning_rate, int size, cudaStream_t stream);

void launch_mse_loss_backward(const float* d_output, const float* d_target,
                              float* d_grad_output, int size, cudaStream_t stream);
                                    
#endif // GPU_KERNELS_OPTIMIZED_2_H
