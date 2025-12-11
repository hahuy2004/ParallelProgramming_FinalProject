#ifndef GPU_KERNELS_OPTIMIZED_2_H
#define GPU_KERNELS_OPTIMIZED_2_H

// Sử dụng fused Conv2D + ReLU + Bias
void launch_conv2d_relu_bias_forward(const float* d_input, float* d_output,
                                     const float* d_weights, const float* d_bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding);

// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ như poolsize =2
void launch_maxpool2d_optimized_forward(const float* d_input, float* d_output,
                                    int batch, int h, int w, int c,
                                    int pool_size, int stride);
                                    
// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ
void launch_conv2d_optimized_backward(const float* d_grad_output, const float* d_input,
                            const float* d_weights, float* d_grad_input,
                            float* d_grad_weights, float* d_grad_bias,
                            int batch, int in_h, int in_w, int in_c,
                            int out_c, int kernel_size, int stride, int padding);                                   
// sử dụng vectorized loads
void launch_relu_backward_optimized(const float* d_grad_output, const float* d_input,
                                    float* d_grad_input, int size);


#endif // GPU_KERNELS_OPTIMIZED_2_H
