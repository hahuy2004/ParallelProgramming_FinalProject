#ifndef GPU_KERNELS_OPTIMIZED_2_H
#define GPU_KERNELS_OPTIMIZED_2_H

// Sử dụng fused Conv2D + ReLU + Bias
void launch_conv2d_forward_relu_fused(const float* d_input, float* d_output,
                                      const float* d_weights, const float* d_bias,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding);

// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ như poolsize =2
void launch_maxpool2d_optimized_forward(const float* d_input, float* d_output, float* indices,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride);
                                    
// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ + fused
void launch_conv2d_relu_backward_fused(const float* d_grad_output, 
                                       const float* d_input,
                                       const float* d_weights, 
                                       const float* d_conv_output,
                                       float* d_grad_input,
                                       float* d_grad_weights, 
                                       float* d_grad_bias,
                                       int batch, int in_h, int in_w, int in_c,
                                       int out_c, int kernel_size, int stride, int padding);
#endif // GPU_KERNELS_OPTIMIZED_2_H
