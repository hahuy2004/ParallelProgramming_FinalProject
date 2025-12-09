#ifndef GPU_KERNELS_OPTIMIZED_1_H
#define GPU_KERNELS_OPTIMIZED_1_H

void launch_conv2d_shared_forward(const float* d_input, float* d_output,
                                   const float* d_weights, const float* d_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding);

void launch_conv2d_shared_backward(const float* d_grad_output, const float* d_input,
                                   const float* d_weights, float* d_grad_input,
                                   float* d_grad_weights, float* d_grad_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding);

#endif // GPU_KERNELS_OPTIMIZED_1_H
