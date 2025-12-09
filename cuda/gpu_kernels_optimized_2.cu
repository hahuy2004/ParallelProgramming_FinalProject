// Tập trung vào Kernel-Level Optimization 
#include "gpu_kernels_optimized_2.h"
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

// ==================== Optimized Fused Conv2D + ReLU + Bias ====================
__global__ void conv2d_relu_bias_kernel(const float* input, float* output,
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
        
        // Use register for accumulation
        float sum = bias[oc];
        
        // Unrolled loops for better performance
        #pragma unroll
        for (int ic = 0; ic < in_c; ++ic) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        int w_idx = oc * in_c * 9 + ic * 9 + kh * 3 + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        // Fused ReLU activation
        output[idx] = fmaxf(0.0f, sum);
    }
}

void launch_conv2d_relu_bias_forward(const float* d_input, float* d_output,
                                     const float* d_weights, const float* d_bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_relu_bias_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// ==================== BACKWARD ====================
// Optimized ReLU Backward with vectorized loads
__global__ void relu_backward_optimized_kernel(const float* grad_output, const float* input,
                                               float* grad_input, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        // Vectorized load (4 floats at once)
        float4 grad_out = reinterpret_cast<const float4*>(grad_output)[idx / 4];
        float4 in_val = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 grad_in;
        
        grad_in.x = (in_val.x > 0.0f) ? grad_out.x : 0.0f;
        grad_in.y = (in_val.y > 0.0f) ? grad_out.y : 0.0f;
        grad_in.z = (in_val.z > 0.0f) ? grad_out.z : 0.0f;
        grad_in.w = (in_val.w > 0.0f) ? grad_out.w : 0.0f;
        
        reinterpret_cast<float4*>(grad_input)[idx / 4] = grad_in;
    } else {
        // Handle remaining elements
        for (int i = idx; i < size; ++i) {
            grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
    }
}

void launch_relu_backward_optimized(const float* d_grad_output, const float* d_input,
                                    float* d_grad_input, int size) {
    int block_size = 256;
    int grid_size = ((size + 3) / 4 + block_size - 1) / block_size;
    
    relu_backward_optimized_kernel<<<grid_size, block_size>>>(d_grad_output, d_input, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}

// Optimized UpSample2D Backward with coalesced memory access
__global__ void upsample2d_backward_optimized_kernel(const float* grad_output, float* grad_input,
                                                     int batch, int in_h, int in_w, int c,
                                                     int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_h * in_w * c;
    
    if (idx < total) {
        int ch = idx % c;
        int iw = (idx / c) % in_w;
        int ih = (idx / c / in_w) % in_h;
        int b = idx / c / in_w / in_h;
        
        float grad_sum = 0.0f;
        
        // Unrolled loop for common scale factors
        if (scale_factor == 2) {
            #pragma unroll
            for (int sh = 0; sh < 2; ++sh) {
                #pragma unroll
                for (int sw = 0; sw < 2; ++sw) {
                    int oh = ih * 2 + sh;
                    int ow = iw * 2 + sw;
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    grad_sum += grad_output[out_idx];
                }
            }
        } else {
            for (int sh = 0; sh < scale_factor; ++sh) {
                for (int sw = 0; sw < scale_factor; ++sw) {
                    int oh = ih * scale_factor + sh;
                    int ow = iw * scale_factor + sw;
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    grad_sum += grad_output[out_idx];
                }
            }
        }
        
        grad_input[idx] = grad_sum;
    }
}

void launch_upsample2d_backward_optimized(const float* d_grad_output, float* d_grad_input,
                                          int batch, int in_h, int in_w, int c,
                                          int scale_factor) {
    int total = batch * in_h * in_w * c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    upsample2d_backward_optimized_kernel<<<grid_size, block_size>>>(
        d_grad_output, d_grad_input, batch, in_h, in_w, c, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}
