// Naive GPU Kernels Implementation
#include "gpu_kernels_naive.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA KERNELS - FORWARD PASS
// ============================================================================

// Conv2D kernel: Basic implementation with configurable stride and padding
__global__ void conv2d_kernel(
    const float* input,   // [C_in, H, W]
    const float* weight,  // [C_out, C_in, K, K]
    const float* bias,    // [C_out]
    float* output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int oc = blockIdx.x;  // Output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Output height
    int ow = blockIdx.z * blockDim.z + threadIdx.z;  // Output width
    
    if (oc >= C_out || oh >= H_out || ow >= W_out) return;
    
    float sum = 0.0f;
    
    // Convolution operation with stride
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                float input_val = 0.0f;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    input_val = input[ic * H_in * W_in + ih * W_in + iw];
                }
                
                int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
                sum += input_val * weight[weight_idx];
            }
        }
    }
    
    sum += bias[oc];
    output[oc * H_out * W_out + oh * W_out + ow] = sum;
}

// ReLU activation kernel
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// MaxPool2D kernel with configurable pool_size and stride
__global__ void maxpool_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H_out, W_out]
    int C, int H, int W,
    int pool_size, int stride)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih_start = oh * stride;
    int iw_start = ow * stride;
    
    // Find max value in the pool_size x pool_size window
    float max_val = -1e38f;  // Very small value
    for (int kh = 0; kh < pool_size; kh++) {
        for (int kw = 0; kw < pool_size; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            if (ih < H && iw < W) {
                float val = input[c * H * W + ih * W + iw];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    output[c * H_out * W_out + oh * W_out + ow] = max_val;
}

// Upsample kernel (nearest neighbor) with configurable scale_factor
__global__ void upsample_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H*scale_factor, W*scale_factor]
    int C, int H, int W,
    int scale_factor)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh / scale_factor;
    int iw = ow / scale_factor;
    
    float val = input[c * H * W + ih * W + iw];
    output[c * H_out * W_out + oh * W_out + ow] = val;
}

// ============================================================================
// CUDA KERNELS - BACKWARD PASS
// ============================================================================

// ReLU backward kernel
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// MaxPool backward kernel with configurable pool_size and stride
__global__ void maxpool_backward_kernel(
    const float* grad_output,  // [C, H_out, W_out]
    const float* input,        // [C, H, W]
    float* grad_input,         // [C, H, W]
    int C, int H, int W,
    int pool_size, int stride)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih_start = oh * stride;
    int iw_start = ow * stride;
    
    // Find which position had the max value
    float max_val = -1e38f;
    int max_i = ih_start, max_j = iw_start;
    
    for (int kh = 0; kh < pool_size; kh++) {
        for (int kw = 0; kw < pool_size; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;
            if (ih < H && iw < W) {
                float val = input[c * H * W + ih * W + iw];
                if (val > max_val) {
                    max_val = val;
                    max_i = ih;
                    max_j = iw;
                }
            }
        }
    }
    
    // Only pass gradient to the max position
    float grad = grad_output[c * H_out * W_out + oh * W_out + ow];
    atomicAdd(&grad_input[c * H * W + max_i * W + max_j], grad);
}

// Upsample backward kernel with configurable scale_factor
__global__ void upsample_backward_kernel(
    const float* grad_output,  // [C, H*scale_factor, W*scale_factor]
    float* grad_input,         // [C, H, W]
    int C, int H, int W,
    int scale_factor)
{
    int c = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c >= C || ih >= H || iw >= W) return;
    
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;
    
    // Sum gradients from all upsampled positions
    float sum = 0.0f;
    for (int kh = 0; kh < scale_factor; kh++) {
        for (int kw = 0; kw < scale_factor; kw++) {
            int oh = ih * scale_factor + kh;
            int ow = iw * scale_factor + kw;
            sum += grad_output[c * H_out * W_out + oh * W_out + ow];
        }
    }
    
    grad_input[c * H * W + ih * W + iw] = sum;
}

// Conv2D weight gradient kernel with configurable parameters
__global__ void conv2d_weight_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* input,        // [C_in, H_in, W_in]
    float* weight_grad,        // [C_out, C_in, kernel_size, kernel_size]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int k_idx = threadIdx.x;  // Flattened kernel index (kh*kernel_size + kw)
    
    if (oc >= C_out || ic >= C_in || k_idx >= kernel_size * kernel_size) return;
    
    int kh = k_idx / kernel_size;
    int kw = k_idx % kernel_size;
    
    float sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                float inp = input[ic * H_in * W_in + ih * W_in + iw];
                sum += grad * inp;
            }
        }
    }
    
    int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
    weight_grad[weight_idx] = sum;
}

// Conv2D bias gradient kernel
__global__ void conv2d_bias_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    float* bias_grad,          // [C_out]
    int C_out, int H_out, int W_out)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= C_out) return;
    
    float sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            sum += grad_output[oc * H_out * W_out + oh * W_out + ow];
        }
    }
    bias_grad[oc] = sum;
}

// Conv2D input gradient kernel with configurable parameters
__global__ void conv2d_input_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* weight,       // [C_out, C_in, kernel_size, kernel_size]
    float* grad_input,         // [C_in, H_in, W_in]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int ic = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ic >= C_in || ih >= H_in || iw >= W_in) return;
    
    float sum = 0.0f;
    for (int oc = 0; oc < C_out; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // For stride > 1, need to check if this input contributes to output
                int oh_temp = ih + padding - kh;
                int ow_temp = iw + padding - kw;
                
                if (oh_temp % stride == 0 && ow_temp % stride == 0) {
                    int oh = oh_temp / stride;
                    int ow = ow_temp / stride;
                    
                    if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                        float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                        int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += grad * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    grad_input[ic * H_in * W_in + ih * W_in + iw] = sum;
}

// ============================================================================
// LOSS AND OPTIMIZATION KERNELS
// ============================================================================

// MSE Loss and gradient kernel
__global__ void mse_loss_kernel(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        grad[idx] = 2.0f * diff / size;
        atomicAdd(loss, diff * diff / size);
    }
}

// Weight update kernel (Simple SGD with gradient clipping)
__global__ void sgd_update_kernel(
    float* weight,
    const float* grad,
    float learning_rate,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        // Check for NaN/Inf first
        if (isnan(g) || isinf(g)) {
            g = 0.0f;
        } else {
            // Clip gradient to prevent explosion
            if (g > 5.0f) g = 5.0f;
            if (g < -5.0f) g = -5.0f;
        }
        weight[idx] -= learning_rate * g;
    }
}

// ============================================================================
// WRAPPER FUNCTIONS (Host-side launch functions)
// ============================================================================

void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    // Grid: (C_out, ceil(H_out/16), ceil(W_out/16))
    // Block: (1, 16, 16)
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C_out, (H_out + 15) / 16, (W_out + 15) / 16);
    
    conv2d_kernel<<<gridDim, blockDim>>>(
        input, weight, bias, output,
        C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_forward(
    const float* input,
    float* output,
    int size)
{
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    relu_kernel<<<numBlocks, threadsPerBlock>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_maxpool_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int pool_size, int stride)
{
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    maxpool_kernel<<<gridDim, blockDim>>>(input, output, C, H, W, pool_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_upsample_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int scale_factor)
{
    int H_out = H * scale_factor;
    int W_out = W * scale_factor;
    
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    upsample_kernel<<<gridDim, blockDim>>>(input, output, C, H, W, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size)
{
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    relu_backward_kernel<<<numBlocks, threadsPerBlock>>>(grad_output, input, grad_input, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_maxpool_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int C, int H, int W,
    int pool_size, int stride)
{
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    maxpool_backward_kernel<<<gridDim, blockDim>>>(grad_output, input, grad_input, C, H, W, pool_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_upsample_backward(
    const float* grad_output,
    float* grad_input,
    int C, int H, int W,
    int scale_factor)
{
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C, (H + 15) / 16, (W + 15) / 16);
    
    upsample_backward_kernel<<<gridDim, blockDim>>>(grad_output, grad_input, C, H, W, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_weight_grad(
    const float* grad_output,
    const float* input,
    float* weight_grad,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    dim3 gridDim(C_out, C_in);
    int threadsPerBlock = kernel_size * kernel_size;  // e.g., 9 for 3x3 kernel
    
    conv2d_weight_grad_kernel<<<gridDim, threadsPerBlock>>>(
        grad_output, input, weight_grad,
        C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_bias_grad(
    const float* grad_output,
    float* bias_grad,
    int C_out, int H_out, int W_out)
{
    int threadsPerBlock = 256;
    int numBlocks = (C_out + threadsPerBlock - 1) / threadsPerBlock;
    
    conv2d_bias_grad_kernel<<<numBlocks, threadsPerBlock>>>(grad_output, bias_grad, C_out, H_out, W_out);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_input_grad(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C_in, (H_in + 15) / 16, (W_in + 15) / 16);
    
    conv2d_input_grad_kernel<<<gridDim, blockDim>>>(
        grad_output, weight, grad_input,
        C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_mse_loss(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    int size)
{
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    mse_loss_kernel<<<numBlocks, threadsPerBlock>>>(pred, target, loss, grad, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_sgd_update(
    float* weight,
    const float* grad,
    float learning_rate,
    int size)
{
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(weight, grad, learning_rate, size);
    
    CUDA_CHECK(cudaGetLastError());
}
