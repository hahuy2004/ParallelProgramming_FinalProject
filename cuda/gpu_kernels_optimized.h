#ifndef GPU_KERNELS_OPTIMIZED_H
#define GPU_KERNELS_OPTIMIZED_H

// Forward declarations for optimized CUDA kernels

// Shared memory optimized convolution
void launch_conv2d_shared_forward(const float* d_input, float* d_output,
                                   const float* d_weights, const float* d_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding);

// Optimized fused Conv2D + ReLU + Bias
void launch_conv2d_relu_bias_forward(const float* d_input, float* d_output,
                                     const float* d_weights, const float* d_bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding);

#endif // GPU_KERNELS_OPTIMIZED_H
