#ifndef GPU_KERNELS_OPTIMIZED_2_H
#define GPU_KERNELS_OPTIMIZED_2_H

//==========================================Forward=========================
// Optimized fused Conv2D + ReLU + Bias
void launch_conv2d_relu_bias_forward(const float* d_input, float* d_output,
                                     const float* d_weights, const float* d_bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding);

//=========================================Backward===========================                               
// Optimized ReLU Backward with vectorized loads
void launch_relu_backward_optimized(const float* d_grad_output, const float* d_input,
                                    float* d_grad_input, int size);


// Optimized UpSample2D Backward
void launch_upsample2d_backward_optimized(const float* d_grad_output, float* d_grad_input,
                                          int batch, int in_h, int in_w, int c,
                                          int scale_factor);

#endif // GPU_KERNELS_OPTIMIZED_2_H
