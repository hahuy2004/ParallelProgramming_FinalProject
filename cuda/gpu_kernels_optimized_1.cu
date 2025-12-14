// Version 1: Tối ưu memory cho kernel - Single image per batch
#include "gpu_kernels_optimized_1.h"
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

#define TILE_SIZE 16
#define CHANNELS_PER_ITER 8

__global__ void conv2d_shared_kernel(const float* input, float* output,
                                      const float* weights, const float* bias,
                                      int in_h, int in_w, int in_c,
                                      int out_c, int out_h, int out_w, 
                                      int kernel_size, int stride, int padding) {    
    const int TILE_WITH_HALO = TILE_SIZE + kernel_size - 1;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int oc = blockIdx.z;  // Output channel
    
    if (oc >= out_c) return;
    
    int out_x = tile_x * TILE_SIZE + tx;
    int out_y = tile_y * TILE_SIZE + ty;
    
    extern __shared__ float s_input[];
    
    float sum = (out_x < out_w && out_y < out_h) ? bias[oc] : 0.0f;
    
    for (int ic_base = 0; ic_base < in_c; ic_base += CHANNELS_PER_ITER) {
        int channels_this_iter = min(CHANNELS_PER_ITER, in_c - ic_base);
        
        int tid = tz * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;
        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        int tile_size = TILE_WITH_HALO * TILE_WITH_HALO * channels_this_iter;
        
        // Load input tile into shared memory
        for (int i = tid; i < tile_size; i += total_threads) {
            int local_c = i / (TILE_WITH_HALO * TILE_WITH_HALO);
            int spatial = i % (TILE_WITH_HALO * TILE_WITH_HALO);
            int local_y = spatial / TILE_WITH_HALO;
            int local_x = spatial % TILE_WITH_HALO;
            
            int in_y = tile_y * TILE_SIZE * stride + local_y - padding;
            int in_x = tile_x * TILE_SIZE * stride + local_x - padding;
            int ic = ic_base + local_c;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                int in_idx = in_y * in_w * in_c + in_x * in_c + ic;
                val = input[in_idx];
            }
            
            s_input[i] = val;
        }
        
        __syncthreads();
        
        // Compute convolution
        if (out_x < out_w && out_y < out_h) {
            for (int ic = 0; ic < channels_this_iter; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int local_y = ty * stride + kh;
                        int local_x = tx * stride + kw;
                        
                        int s_idx = ic * TILE_WITH_HALO * TILE_WITH_HALO + 
                                   local_y * TILE_WITH_HALO + local_x;
                        
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   (ic_base + ic) * kernel_size * kernel_size + 
                                   kh * kernel_size + kw;
                        
                        sum += s_input[s_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (out_x < out_w && out_y < out_h) {
        int out_idx = out_y * out_w * out_c + out_x * out_c + oc;
        output[out_idx] = sum;
    }
}

void launch_conv2d_shared_forward(const float* d_input, float* d_output,
                                   const float* d_weights, const float* d_bias,
                                   int in_h, int in_w, int in_c,
                                   int out_c, int out_h, int out_w,
                                   int kernel_size, int stride, int padding) {
    
    const int TILE_WITH_HALO = TILE_SIZE + kernel_size - 1;
    
    int shared_mem_size = TILE_WITH_HALO * TILE_WITH_HALO * CHANNELS_PER_ITER * sizeof(float);
    
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    
    int grid_x = (out_w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (out_h + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = out_c;
    dim3 grid(grid_x, grid_y, grid_z);
    
    conv2d_shared_kernel<<<grid, block, shared_mem_size>>>(
        d_input, d_output, d_weights, d_bias,
        in_h, in_w, in_c, out_c, out_h, out_w, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void conv2d_shared_backward_kernel(const float* grad_output, const float* input,
                                               const float* weights, float* grad_input,
                                               float* grad_weights, float* grad_bias,
                                               int in_h, int in_w, int in_c, 
                                               int out_c, int out_h, int out_w,  
                                               int kernel_size, int stride, int padding) {
    const int TILE_WITH_HALO = TILE_SIZE + kernel_size - 1;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int oc = blockIdx.z;  // Output channel
    
    if (oc >= out_c) return;
    
    int out_x = tile_x * TILE_SIZE + tx;
    int out_y = tile_y * TILE_SIZE + ty;
    
    extern __shared__ float shared_mem[];
    float* s_grad_out = shared_mem;
    float* s_input = s_grad_out + TILE_SIZE * TILE_SIZE;
    float* s_grad_bias_local = s_input + TILE_WITH_HALO * TILE_WITH_HALO * CHANNELS_PER_ITER;
    
    // Load gradient output
    float grad_val = 0.0f;
    if (out_x < out_w && out_y < out_h) {
        int grad_idx = out_y * out_w * out_c + out_x * out_c + oc;
        grad_val = grad_output[grad_idx];
    }
    s_grad_out[ty * TILE_SIZE + tx] = grad_val;
    
    if (tx == 0 && ty == 0) {
        s_grad_bias_local[0] = 0.0f;
    }
    __syncthreads();
    
    // Accumulate bias gradient
    atomicAdd(s_grad_bias_local, grad_val);
    
    // Process input channels in chunks
    for (int ic_base = 0; ic_base < in_c; ic_base += CHANNELS_PER_ITER) {
        int channels_this_iter = min(CHANNELS_PER_ITER, in_c - ic_base);
        
        int tid = ty * blockDim.x + tx;
        int total_threads = blockDim.x * blockDim.y;
        int tile_size = TILE_WITH_HALO * TILE_WITH_HALO * channels_this_iter;
        
        // Load input tile
        for (int i = tid; i < tile_size; i += total_threads) {
            int local_c = i / (TILE_WITH_HALO * TILE_WITH_HALO);
            int spatial = i % (TILE_WITH_HALO * TILE_WITH_HALO);
            int local_y = spatial / TILE_WITH_HALO;
            int local_x = spatial % TILE_WITH_HALO;
            
            int in_y = tile_y * TILE_SIZE * stride + local_y - padding;
            int in_x = tile_x * TILE_SIZE * stride + local_x - padding;
            int ic = ic_base + local_c;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                int in_idx = in_y * in_w * in_c + in_x * in_c + ic;
                val = input[in_idx];
            }
            
            s_input[i] = val;
        }
        
        __syncthreads();
        
        // Compute gradients
        if (out_x < out_w && out_y < out_h) {
            for (int ic = 0; ic < channels_this_iter; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int local_y = ty * stride + kh;
                        int local_x = tx * stride + kw;
                        int in_y = out_y * stride - padding + kh;
                        int in_x = out_x * stride - padding + kw;
                        
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            int s_idx = ic * TILE_WITH_HALO * TILE_WITH_HALO + 
                                       local_y * TILE_WITH_HALO + local_x;
                            float in_val = s_input[s_idx];
                            
                            int w_idx = oc * in_c * kernel_size * kernel_size + 
                                       (ic_base + ic) * kernel_size * kernel_size + 
                                       kh * kernel_size + kw;
                            
                            // Gradient w.r.t weights
                            atomicAdd(&grad_weights[w_idx], in_val * grad_val);
                            
                            // Gradient w.r.t input
                            int in_idx = in_y * in_w * in_c + in_x * in_c + (ic_base + ic);
                            atomicAdd(&grad_input[in_idx], weights[w_idx] * grad_val);
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write bias gradient
    if (tx == 0 && ty == 0) {
        atomicAdd(&grad_bias[oc], s_grad_bias_local[0]);
    }
}

void launch_conv2d_shared_backward(const float* d_grad_output, const float* d_input,
                                    const float* d_weights, float* d_grad_input,
                                    float* d_grad_weights, float* d_grad_bias,
                                    int in_h, int in_w, int in_c,
                                    int out_c, int out_h, int out_w,
                                    int kernel_size, int stride, int padding) {    
    const int TILE_WITH_HALO = TILE_SIZE + kernel_size - 1;
    
    // Shared memory: grad_output tile + input tile + bias accumulator
    int shared_mem_size = (TILE_SIZE * TILE_SIZE + 
                          TILE_WITH_HALO * TILE_WITH_HALO * CHANNELS_PER_ITER + 
                          1) * sizeof(float);
    
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    int grid_x = (out_w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (out_h + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = out_c;
    dim3 grid(grid_x, grid_y, grid_z);
    
    conv2d_shared_backward_kernel<<<grid, block, shared_mem_size>>>(
        d_grad_output, d_input, d_weights, d_grad_input,
        d_grad_weights, d_grad_bias,
        in_h, in_w, in_c, out_c, out_h, out_w, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}