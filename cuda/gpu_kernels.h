#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

// Forward declarations of CUDA kernels

// Convolution kernel
void launch_conv2d_forward(const float* d_input, float* d_output,
                           const float* d_weights, const float* d_bias,
                           int batch, int in_h, int in_w, int in_c,
                           int out_c, int kernel_size, int stride, int padding);

// ReLU activation
void launch_relu_forward(float* d_data, int size);

// Max pooling
void launch_maxpool2d_forward(const float* d_input, float* d_output, float* indices, 
                              int batch, int h, int w, int c,
                              int pool_size, int stride);

// Upsampling
void launch_upsample2d_forward(const float* d_input, float* d_output,
                               int batch, int in_h, int in_w, int c,
                               int scale_factor);

// MSE Loss computation
void launch_mse_loss(const float* d_input, const float* d_output,
                     float* d_loss, int size);

// Gradient initialization
void launch_zero_grad(float* d_grad, int size);

// Weight update (SGD)
void launch_sgd_update(float* d_weights, const float* d_grad,
                       float learning_rate, int size);

// Fused operations (optimized version)
void launch_conv2d_relu_forward(const float* d_input, float* d_output,
                                const float* d_weights, const float* d_bias,
                                int batch, int in_h, int in_w, int in_c,
                                int out_c, int kernel_size, int stride, int padding);

// ==================== BACKWARD PASS FUNCTIONS ====================

// Conv2D Backward
void launch_conv2d_backward(const float* d_grad_output, const float* d_input,
                            const float* d_weights, float* d_grad_input,
                            float* d_grad_weights, float* d_grad_bias,
                            int batch, int in_h, int in_w, int in_c,
                            int out_c, int kernel_size, int stride, int padding);

// ReLU Backward
void launch_relu_backward(const float* d_grad_output, const float* d_input,
                         float* d_grad_input, int size);

// MaxPool2D Backward
void launch_maxpool2d_backward(const float* d_grad_output, float* d_input, const float* indices,
                               const float* d_output, float* d_grad_input,
                               int batch, int h, int w, int c,
                               int pool_size, int stride);

// UpSample2D Backward
void launch_upsample2d_backward(const float* d_grad_output, float* d_grad_input,
                                int batch, int in_h, int in_w, int c,
                                int scale_factor);

// MSE Loss Backward
void launch_mse_loss_backward(const float* d_output, const float* d_target,
                              float* d_grad_output, int size);

#endif // GPU_KERNELS_H
