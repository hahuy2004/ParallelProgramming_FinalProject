// Optimized GPU Kernels with Shared Memory - Version 1
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

// ============================================================================
// SHARED MEMORY CONV2D FORWARD
// ============================================================================

#define TILE_SIZE 16

__global__ void conv2d_shared_forward_kernel(
    const float* input,   // [C_in, H_in, W_in]
    const float* weight,  // [C_out, C_in, K, K]
    const float* bias,    // [C_out]
    float* output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    // Shared memory for weights of current output channel
    extern __shared__ float shared_mem[];
    float* s_weight = shared_mem;  // size: C_in * kernel_size * kernel_size
    
    int oc = blockIdx.x;  // Output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (oc >= C_out) return;
    
    // Load weights for this output channel into shared memory
    int tid = threadIdx.y * blockDim.z + threadIdx.z;
    int threads_per_block = blockDim.y * blockDim.z;
    int weight_size = C_in * kernel_size * kernel_size;
    
    for (int i = tid; i < weight_size; i += threads_per_block) {
        int ic = i / (kernel_size * kernel_size);
        int k_offset = i % (kernel_size * kernel_size);
        int kh = k_offset / kernel_size;
        int kw = k_offset % kernel_size;
        int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
        s_weight[i] = weight[weight_idx];
    }
    
    __syncthreads();
    
    // Compute convolution
    if (oh < H_out && ow < W_out) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    
                    float input_val = 0.0f;
                    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                        input_val = input[ic * H_in * W_in + ih * W_in + iw];
                    }
                    
                    int s_weight_idx = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input_val * s_weight[s_weight_idx];
                }
            }
        }
        
        sum += bias[oc];
        output[oc * H_out * W_out + oh * W_out + ow] = sum;
    }
}

void launch_conv2d_shared_forward(
    const float* d_input, 
    float* d_output,
    const float* d_weights, 
    const float* d_bias,
    int in_h, int in_w, int in_c,
    int out_c, int out_h, int out_w,
    int kernel_size, int stride, int padding)
{
    dim3 block(1, TILE_SIZE, TILE_SIZE);
    dim3 grid(out_c, (out_h + TILE_SIZE - 1) / TILE_SIZE, (out_w + TILE_SIZE - 1) / TILE_SIZE);
    
    int shared_mem_size = in_c * kernel_size * kernel_size * sizeof(float);
    
    conv2d_shared_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_input, d_weights, d_bias, d_output,
        in_c, in_h, in_w, out_c, out_h, out_w,
        kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// SHARED MEMORY CONV2D BACKWARD
// ============================================================================

__global__ void conv2d_shared_weight_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* input,        // [C_in, H_in, W_in]
    float* grad_weight,        // [C_out, C_in, K, K]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int kh = threadIdx.y;
    int kw = threadIdx.z;
    
    if (oc >= C_out || ic >= C_in || kh >= kernel_size || kw >= kernel_size) return;
    
    float grad_sum = 0.0f;
    
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float grad_out_val = grad_output[oc * H_out * W_out + oh * W_out + ow];
                float input_val = input[ic * H_in * W_in + ih * W_in + iw];
                grad_sum += grad_out_val * input_val;
            }
        }
    }
    
    int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
    grad_weight[weight_idx] = grad_sum;
}

__global__ void conv2d_shared_bias_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    float* grad_bias,          // [C_out]
    int C_out, int H_out, int W_out)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oc >= C_out) return;
    
    float grad_sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            grad_sum += grad_output[oc * H_out * W_out + oh * W_out + ow];
        }
    }
    
    grad_bias[oc] = grad_sum;
}

__global__ void conv2d_shared_input_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* weight,       // [C_out, C_in, K, K]
    float* grad_input,         // [C_in, H_in, W_in]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    // Shared memory for weights of current input channel
    extern __shared__ float s_weight[];  // size: C_out * kernel_size * kernel_size
    
    int ic = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ic >= C_in) return;
    
    // Load weights for this input channel
    int tid = threadIdx.y * blockDim.z + threadIdx.z;
    int threads_per_block = blockDim.y * blockDim.z;
    int weight_size = C_out * kernel_size * kernel_size;
    
    for (int i = tid; i < weight_size; i += threads_per_block) {
        int oc = i / (kernel_size * kernel_size);
        int k_offset = i % (kernel_size * kernel_size);
        int kh = k_offset / kernel_size;
        int kw = k_offset % kernel_size;
        int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
        s_weight[i] = weight[weight_idx];
    }
    
    __syncthreads();
    
    // Compute gradient
    if (ih < H_in && iw < W_in) {
        float grad_sum = 0.0f;
        
        for (int oc = 0; oc < C_out; oc++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int oh_base = ih + padding - kh;
                    int ow_base = iw + padding - kw;
                    
                    if (oh_base >= 0 && oh_base % stride == 0 && ow_base >= 0 && ow_base % stride == 0) {
                        int oh = oh_base / stride;
                        int ow = ow_base / stride;
                        
                        if (oh < H_out && ow < W_out) {
                            float grad_out_val = grad_output[oc * H_out * W_out + oh * W_out + ow];
                            int s_weight_idx = oc * kernel_size * kernel_size + kh * kernel_size + kw;
                            grad_sum += grad_out_val * s_weight[s_weight_idx];
                        }
                    }
                }
            }
        }
        
        grad_input[ic * H_in * W_in + ih * W_in + iw] = grad_sum;
    }
}

void launch_conv2d_shared_backward(
    const float* d_grad_output, 
    const float* d_input,
    const float* d_weights, 
    float* d_grad_input,
    float* d_grad_weights, 
    float* d_grad_bias,
    int in_h, int in_w, int in_c,
    int out_c, int out_h, int out_w,
    int kernel_size, int stride, int padding)
{
    // Compute weight gradients
    {
        dim3 block(1, kernel_size, kernel_size);
        dim3 grid(out_c, in_c, 1);
        
        conv2d_shared_weight_grad_kernel<<<grid, block>>>(
            d_grad_output, d_input, d_grad_weights,
            in_c, in_h, in_w, out_c, out_h, out_w,
            kernel_size, stride, padding);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Compute bias gradients
    {
        int block_size = 256;
        int grid_size = (out_c + block_size - 1) / block_size;
        
        conv2d_shared_bias_grad_kernel<<<grid_size, block_size>>>(
            d_grad_output, d_grad_bias, out_c, out_h, out_w);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Compute input gradients
    {
        dim3 block(1, TILE_SIZE, TILE_SIZE);
        dim3 grid(in_c, (in_h + TILE_SIZE - 1) / TILE_SIZE, (in_w + TILE_SIZE - 1) / TILE_SIZE);
        
        int shared_mem_size = out_c * kernel_size * kernel_size * sizeof(float);
        
        conv2d_shared_input_grad_kernel<<<grid, block, shared_mem_size>>>(
            d_grad_output, d_weights, d_grad_input,
            in_c, in_h, in_w, out_c, out_h, out_w,
            kernel_size, stride, padding);
        
        CUDA_CHECK(cudaGetLastError());
    }
}
