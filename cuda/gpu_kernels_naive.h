#ifndef GPU_KERNELS_NAIVE_H
#define GPU_KERNELS_NAIVE_H

// ============================================================================
// FORWARD PASS KERNELS
// ============================================================================

// Conv2D forward kernel
void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);

// ReLU forward kernel
void launch_relu_forward(
    const float* input,
    float* output,
    int size);

// MaxPool2D forward kernel
void launch_maxpool_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int pool_size, int stride);

// Upsample forward kernel
void launch_upsample_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int scale_factor);

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

// ReLU backward kernel
void launch_relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size);

// MaxPool backward kernel
void launch_maxpool_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int C, int H, int W,
    int pool_size, int stride);

// Upsample backward kernel
void launch_upsample_backward(
    const float* grad_output,
    float* grad_input,
    int C, int H, int W,
    int scale_factor);

// Conv2D weight gradient kernel
void launch_conv2d_weight_grad(
    const float* grad_output,
    const float* input,
    float* weight_grad,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);

// Conv2D bias gradient kernel
void launch_conv2d_bias_grad(
    const float* grad_output,
    float* bias_grad,
    int C_out, int H_out, int W_out);

// Conv2D input gradient kernel
void launch_conv2d_input_grad(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);

// ============================================================================
// LOSS AND OPTIMIZATION KERNELS
// ============================================================================

// MSE Loss and gradient kernel
void launch_mse_loss(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    int size);

// SGD weight update kernel
void launch_sgd_update(
    float* weight,
    const float* grad,
    float learning_rate,
    int size);

#endif // GPU_KERNELS_NAIVE_H
