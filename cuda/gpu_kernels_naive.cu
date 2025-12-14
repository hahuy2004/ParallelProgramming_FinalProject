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

// Conv2D kernel: Basic implementation
__global__ void conv2d_kernel(
    const float* input,   // [C_in, H, W]
    const float* weight,  // [C_out, C_in, K, K]
    const float* bias,    // [C_out]
    float* output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int oc = blockIdx.x;  // Output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Output height
    int ow = blockIdx.z * blockDim.z + threadIdx.z;  // Output width
    
    if (oc >= C_out || oh >= H_out || ow >= W_out) return;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;
                
                float input_val = 0.0f;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    input_val = input[ic * H_in * W_in + ih * W_in + iw];
                }
                
                int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
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

// MaxPool2D kernel (2x2, stride 2)
__global__ void maxpool_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H/2, W/2]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H / 2;
    int W_out = W / 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    float max_val = input[c * H * W + ih * W + iw];
    max_val = fmaxf(max_val, input[c * H * W + ih * W + (iw + 1)]);
    max_val = fmaxf(max_val, input[c * H * W + (ih + 1) * W + iw]);
    max_val = fmaxf(max_val, input[c * H * W + (ih + 1) * W + (iw + 1)]);
    
    output[c * H_out * W_out + oh * W_out + ow] = max_val;
}

// Upsample2x kernel (nearest neighbor)
__global__ void upsample_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H*2, W*2]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh / 2;
    int iw = ow / 2;
    
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

// MaxPool backward kernel
__global__ void maxpool_backward_kernel(
    const float* grad_output,  // [C, H/2, W/2]
    const float* input,        // [C, H, W]
    float* grad_input,         // [C, H, W]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H / 2;
    int W_out = W / 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    // Find which position had the max value
    float max_val = input[c * H * W + ih * W + iw];
    int max_i = ih, max_j = iw;
    
    float val = input[c * H * W + ih * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_i = ih; max_j = iw + 1; }
    
    val = input[c * H * W + (ih + 1) * W + iw];
    if (val > max_val) { max_val = val; max_i = ih + 1; max_j = iw; }
    
    val = input[c * H * W + (ih + 1) * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_i = ih + 1; max_j = iw + 1; }
    
    // Only pass gradient to the max position
    float grad = grad_output[c * H_out * W_out + oh * W_out + ow];
    atomicAdd(&grad_input[c * H * W + max_i * W + max_j], grad);
}

// Upsample backward kernel
__global__ void upsample_backward_kernel(
    const float* grad_output,  // [C, H*2, W*2]
    float* grad_input,         // [C, H, W]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c >= C || ih >= H || iw >= W) return;
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    // Sum gradients from all 4 upsampled positions
    float sum = 0.0f;
    sum += grad_output[c * H_out * W_out + (ih * 2) * W_out + (iw * 2)];
    sum += grad_output[c * H_out * W_out + (ih * 2) * W_out + (iw * 2 + 1)];
    sum += grad_output[c * H_out * W_out + (ih * 2 + 1) * W_out + (iw * 2)];
    sum += grad_output[c * H_out * W_out + (ih * 2 + 1) * W_out + (iw * 2 + 1)];
    
    grad_input[c * H * W + ih * W + iw] = sum;
}

// Conv2D weight gradient kernel
__global__ void conv2d_weight_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* input,        // [C_in, H_in, W_in]
    float* weight_grad,        // [C_out, C_in, K, K]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int k_idx = threadIdx.x;  // Flattened kernel index (kh*K + kw)
    
    if (oc >= C_out || ic >= C_in || k_idx >= K * K) return;
    
    int kh = k_idx / K;
    int kw = k_idx % K;
    
    float sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int ih = oh + kh - pad;
            int iw = ow + kw - pad;
            
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                float inp = input[ic * H_in * W_in + ih * W_in + iw];
                sum += grad * inp;
            }
        }
    }
    
    int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
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

// Conv2D input gradient kernel
__global__ void conv2d_input_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* weight,       // [C_out, C_in, K, K]
    float* grad_input,         // [C_in, H_in, W_in]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int ic = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ic >= C_in || ih >= H_in || iw >= W_in) return;
    
    float sum = 0.0f;
    for (int oc = 0; oc < C_out; oc++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int oh = ih - kh + pad;
                int ow = iw - kw + pad;
                
                if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                    float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                    int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
                    sum += grad * weight[weight_idx];
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
    int K, int pad)
{
    // Grid: (C_out, ceil(H_out/16), ceil(W_out/16))
    // Block: (1, 16, 16)
    dim3 threads(1, 16, 16);
    dim3 blocks(C_out, (H_out + 15) / 16, (W_out + 15) / 16);
    
    conv2d_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        C_in, H_in, W_in, C_out, H_out, W_out, K, pad);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_forward(
    const float* input,
    float* output,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_maxpool_forward(
    const float* input,
    float* output,
    int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    dim3 threads(1, 16, 16);
    dim3 blocks(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    maxpool_kernel<<<blocks, threads>>>(input, output, C, H, W);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_upsample_forward(
    const float* input,
    float* output,
    int C, int H, int W)
{
    int H_out = H * 2;
    int W_out = W * 2;
    
    dim3 threads(1, 16, 16);
    dim3 blocks(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    upsample_kernel<<<blocks, threads>>>(input, output, C, H, W);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_maxpool_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    dim3 threads(1, 16, 16);
    dim3 blocks(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    maxpool_backward_kernel<<<blocks, threads>>>(grad_output, input, grad_input, C, H, W);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_upsample_backward(
    const float* grad_output,
    float* grad_input,
    int C, int H, int W)
{
    dim3 threads(1, 16, 16);
    dim3 blocks(C, (H + 15) / 16, (W + 15) / 16);
    
    upsample_backward_kernel<<<blocks, threads>>>(grad_output, grad_input, C, H, W);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_weight_grad(
    const float* grad_output,
    const float* input,
    float* weight_grad,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    dim3 blocks(C_out, C_in);
    int threads = K * K;  // 9 for 3x3 kernel
    
    conv2d_weight_grad_kernel<<<blocks, threads>>>(
        grad_output, input, weight_grad,
        C_in, H_in, W_in, C_out, H_out, W_out, K, pad);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_bias_grad(
    const float* grad_output,
    float* bias_grad,
    int C_out, int H_out, int W_out)
{
    int threads = 256;
    int blocks = (C_out + threads - 1) / threads;
    
    conv2d_bias_grad_kernel<<<blocks, threads>>>(grad_output, bias_grad, C_out, H_out, W_out);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv2d_input_grad(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    dim3 threads(1, 16, 16);
    dim3 blocks(C_in, (H_in + 15) / 16, (W_in + 15) / 16);
    
    conv2d_input_grad_kernel<<<blocks, threads>>>(
        grad_output, weight, grad_input,
        C_in, H_in, W_in, C_out, H_out, W_out, K, pad);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_mse_loss(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    mse_loss_kernel<<<blocks, threads>>>(pred, target, loss, grad, size);
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_sgd_update(
    float* weight,
    const float* grad,
    float learning_rate,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    sgd_update_kernel<<<blocks, threads>>>(weight, grad, learning_rate, size);
    
    CUDA_CHECK(cudaGetLastError());
}
