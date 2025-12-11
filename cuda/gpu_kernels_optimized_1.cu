// Version 1: Tập trung vào tối ưu memory cho các kernel
#include "gpu_kernels_optimized_1.h"
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

// Tile size for shared memory
#define TILE_SIZE 8

// ==================== Shared Memory Optimized Convolution ====================
__global__ void conv2d_shared_kernel(const float* input, float* output,
                                     const float* weights, const float* bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding) {
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2][8];  // Tile with padding for 3x3 kernel
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Flatten batch and output channel into single dimension
    int batch_oc = blockIdx.z;
    int b = batch_oc / out_c;
    int oc = batch_oc % out_c;
    
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    if (out_x < out_w && out_y < out_h && oc < out_c && b < batch) {
        // Collaborative loading into shared memory
        for (int ic = tz; ic < min(8, in_c); ic += blockDim.z) {
            int in_x = out_x * stride - padding + tx;
            int in_y = out_y * stride - padding + ty;
            
            if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h) {
                int in_idx = b * in_h * in_w * in_c + in_y * in_w * in_c + in_x * in_c + ic;
                tile[ty][tx][ic] = input[in_idx];
            } else {
                tile[ty][tx][ic] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        float sum = bias[oc];
        
        for (int ic = 0; ic < min(8, in_c); ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    if (ty + kh < TILE_SIZE + 2 && tx + kw < TILE_SIZE + 2) {
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += tile[ty + kh][tx + kw][ic] * weights[w_idx];
                    }
                }
            }
        }
        
        int out_idx = b * out_h * out_w * out_c + out_y * out_w * out_c + out_x * out_c + oc;
        output[out_idx] = sum;
    }
}

void launch_conv2d_shared_forward(const float* d_input, float* d_output,
                                   const float* d_weights, const float* d_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE, 4);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * out_c);  // Flatten batch and out_c
    
    conv2d_shared_kernel<<<grid, block>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// Sử dụng shared memory cho giai đoạn backward
__global__ void conv2d_shared_backward_kernel(const float* grad_output, const float* input,
                                              const float* weights, float* grad_input,
                                              float* grad_weights, float* grad_bias,
                                              int batch, int in_h, int in_w, int in_c,
                                              int out_c, int kernel_size, int stride, int padding) {
    __shared__ float tile_input[TILE_SIZE + 2][TILE_SIZE + 2][8];
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int batch_oc = blockIdx.z;
    int b = batch_oc / out_c;
    int oc = batch_oc % out_c;
    
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    if (out_x < out_w && out_y < out_h && oc < out_c && b < batch) {
        // Load gradient output into shared memory
        int out_idx = b * out_h * out_w * out_c + out_y * out_w * out_c + out_x * out_c + oc;
        float grad_out_val = grad_output[out_idx];
        
        // Accumulate gradient for bias
        if (tx == 0 && ty == 0) {
            atomicAdd(&grad_bias[oc], grad_out_val);
        }
        
        // Load input into shared memory
        for (int ic = tz; ic < min(8, in_c); ic += blockDim.z) {
            int in_x = out_x * stride - padding + tx;
            int in_y = out_y * stride - padding + ty;
            
            if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h) {
                int in_idx = b * in_h * in_w * in_c + in_y * in_w * in_c + in_x * in_c + ic;
                tile_input[ty][tx][ic] = input[in_idx];
            } else {
                tile_input[ty][tx][ic] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute gradients
        for (int ic = 0; ic < min(8, in_c); ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = out_y * stride - padding + kh;
                    int iw = out_x * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        // Gradient w.r.t. weights
                        if (ty + kh < TILE_SIZE + 2 && tx + kw < TILE_SIZE + 2) {
                            atomicAdd(&grad_weights[w_idx], tile_input[ty + kh][tx + kw][ic] * grad_out_val);
                        }
                        
                        // Gradient w.r.t. input
                        atomicAdd(&grad_input[in_idx], weights[w_idx] * grad_out_val);
                    }
                }
            }
        }
    }
}

void launch_conv2d_shared_backward(const float* d_grad_output, const float* d_input,
                                   const float* d_weights, float* d_grad_input,
                                   float* d_grad_weights, float* d_grad_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    dim3 block(TILE_SIZE, TILE_SIZE, 4);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * out_c);
    
    conv2d_shared_backward_kernel<<<grid, block>>>(
        d_grad_output, d_input, d_weights, d_grad_input,
        d_grad_weights, d_grad_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}