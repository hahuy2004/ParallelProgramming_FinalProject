#include "gpu_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== Convolution Kernel ====================
__global__ void conv2d_forward_kernel(const float* input, float* output,
                                      const float* weights, const float* bias,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx < total) {
        int oc = idx % out_c;
        int ow = (idx / out_c) % out_w;
        int oh = (idx / out_c / out_w) % out_h;
        int b = idx / out_c / out_w / out_h;
        
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

void launch_conv2d_forward(const float* d_input, float* d_output,
                           const float* d_weights, const float* d_bias,
                           int batch, int in_h, int in_w, int in_c,
                           int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// ==================== ReLU Kernel ====================
__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void launch_relu_forward(float* d_data, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_forward_kernel<<<grid_size, block_size>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}

// ==================== Max Pooling Kernel ====================
__global__ void maxpool2d_forward_kernel(const float* input, float* output,
                                         int batch, int h, int w, int c,
                                         int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;
    
    if (idx < total) {
        int ch = idx % c;
        int ow = (idx / c) % out_w;
        int oh = (idx / c / out_w) % out_h;
        int b = idx / c / out_w / out_h;
        
        float max_val = -INFINITY;
        
        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                int in_idx = b * h * w * c + ih * w * c + iw * c + ch;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        
        output[idx] = max_val;
    }
}

void launch_maxpool2d_forward(const float* d_input, float* d_output,
                              int batch, int h, int w, int c,
                              int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    maxpool2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch, h, w, c, pool_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
}

// ==================== Upsampling Kernel ====================
__global__ void upsample2d_forward_kernel(const float* input, float* output,
                                          int batch, int in_h, int in_w, int c,
                                          int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * c;
    
    if (idx < total) {
        int ch = idx % c;
        int ow = (idx / c) % out_w;
        int oh = (idx / c / out_w) % out_h;
        int b = idx / c / out_w / out_h;
        
        int ih = oh / scale_factor;
        int iw = ow / scale_factor;
        int in_idx = b * in_h * in_w * c + ih * in_w * c + iw * c + ch;
        
        output[idx] = input[in_idx];
    }
}

void launch_upsample2d_forward(const float* d_input, float* d_output,
                               int batch, int in_h, int in_w, int c,
                               int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    int total = batch * out_h * out_w * c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    upsample2d_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, batch, in_h, in_w, c, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

// ==================== MSE Loss Kernel ====================
__global__ void mse_loss_kernel(const float* input, const float* output,
                                float* partial_sums, int size) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < size) {
        float diff = output[idx] - input[idx];
        sum = diff * diff;
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(partial_sums, shared_sum[0]);
    }
}

void launch_mse_loss(const float* d_input, const float* d_output,
                     float* d_loss, int size) {
    // Initialize loss to 0
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    mse_loss_kernel<<<grid_size, block_size>>>(d_input, d_output, d_loss, size);
    CUDA_CHECK(cudaGetLastError());
}

// ==================== Utility Kernels ====================
__global__ void zero_grad_kernel(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

void launch_zero_grad(float* d_grad, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    zero_grad_kernel<<<grid_size, block_size>>>(d_grad, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sgd_update_kernel(float* weights, const float* grad,
                                  float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad[idx];
    }
}

void launch_sgd_update(float* d_weights, const float* d_grad,
                       float learning_rate, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sgd_update_kernel<<<grid_size, block_size>>>(d_weights, d_grad, learning_rate, size);
    CUDA_CHECK(cudaGetLastError());
}

// ==================== Fused Convolution + ReLU (Optimized) ====================
__global__ void conv2d_relu_forward_kernel(const float* input, float* output,
                                           const float* weights, const float* bias,
                                           int batch, int in_h, int in_w, int in_c,
                                           int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx < total) {
        int oc = idx % out_c;
        int ow = (idx / out_c) % out_w;
        int oh = (idx / out_c / out_w) % out_h;
        int b = idx / out_c / out_w / out_h;
        
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        // Fused ReLU activation
        output[idx] = fmaxf(0.0f, sum);
    }
}

void launch_conv2d_relu_forward(const float* d_input, float* d_output,
                                const float* d_weights, const float* d_bias,
                                int batch, int in_h, int in_w, int in_c,
                                int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_relu_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}
