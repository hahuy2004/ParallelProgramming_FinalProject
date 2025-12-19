#ifndef GPU_KERNELS_OPTIMIZED_2_H
#define GPU_KERNELS_OPTIMIZED_2_H

// Sử dụng fused Conv2D + ReLU + Bias
void launch_conv2d_relu_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);

// Sử dụng unroll looping cho các vòng duyệt kernel kích thước nhỏ như poolsize =2
void launch_maxpool_unroll_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int pool_size, int stride);
                                    
void launch_maxpool_unroll_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int C, int H, int W,
    int pool_size, int stride);

void launch_conv2d_input_grad_unroll(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);
#endif // GPU_KERNELS_OPTIMIZED_2_H
