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

// Forward
__global__ void conv2d_forward_relu_fused(const float* input, float* output,
                                          const float* weights, const float* bias,
                                          int batch, int in_h, int in_w, int in_c, int out_h, int out_w,
                                          int out_c, int kernel_size, int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx >= total) return;
    
    int oc = idx % out_c;
    int ow = (idx / out_c) % out_w;
    int oh = (idx / out_c / out_w) % out_h;
    int b = idx / out_c / out_w / out_h;
    
    float sum = bias[oc];
    
    // Fused convolution with loop unrolling for kernel_size=3
    if (kernel_size == 3){
        for (int ic = 0; ic < in_c; ++ic) {
            // Unroll only small kernel dimensions
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh * stride - padding + kh;
                if (ih >= 0 && ih < in_h) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow * stride - padding + kw;
                        if (iw >= 0 && iw < in_w) {
                            int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                            int w_idx = oc * in_c * 9 + ic * 9 + kh * 3 + kw;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
            }
        }
    }
    else{
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih = oh * stride - padding + kh;
                if (ih >= 0 && ih < in_h) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int iw = ow * stride - padding + kw;
                        if (iw >= 0 && iw < in_w) {
                            int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                            int w_idx = oc * in_c * 9 + ic * 9 + kh * 3 + kw;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
            }
        }
    }

    // Fused ReLU activation: max(0, sum)
    output[idx] = fmaxf(0.0f, sum);
}

void launch_conv2d_forward_relu_fused(const float* d_input, float* d_output,
                                      const float* d_weights, const float* d_bias,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_forward_relu_fused<<<grid_size, block_size>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// Hàm này áp dụng cho pool size = 2 theo đề bài quy định
__global__ void maxpool2d_forward_optimized_kernel(const float* input, float* output, float* indices,
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
        int max_idx = -1;
        if (pool_size == 2 && stride == 2) {

            // Loop unrolling cho trường hợp pool size = 2
            // Unroll hoàn toàn cho 2x2 pooling với stride=2
            // Loại bỏ loop overhead, compiler có thể optimize tốt hơn
            int ih0 = oh * stride;
            int iw0 = ow * stride;
            int base_idx = b * h * w * c + ih0 * w * c + iw0 * c + ch;
            
            // Load 4 giá trị và tìm max (fully unrolled)
            float val0 = input[base_idx];
            float val1 = input[base_idx + c];
            float val2 = input[base_idx + w * c];
            float val3 = input[base_idx + w * c + c];
            
            max_val = val0;
            max_idx = base_idx;

            // So sánh với val1
            if (val1 > max_val) {
                max_val = val1;
                max_idx = base_idx + c;
            }
            // So sánh với val2
            if (val2 > max_val) {
                max_val = val2;
                max_idx = base_idx + w * c;
            }
            // So sánh với val3
            if (val3 > max_val) {
                max_val = val3;
                max_idx = base_idx + w * c + c;
            }
        }
        else{
            for (int ph = 0; ph < pool_size; ++ph) {
                for (int pw = 0; pw < pool_size; ++pw) {
                    int ih = oh * stride + ph;
                    int iw = ow * stride + pw;
                    int in_idx = b * h * w * c + ih * w * c + iw * c + ch;
                    max_val = fmaxf(max_val, input[in_idx]);
                    if(input[in_idx] > max_val){
                        max_val = input[in_idx];
                        max_idx = in_idx;
                    }
                }
            }
        }
        indices[idx] = (float)max_idx;
        output[idx] = max_val;
    }
}

void launch_maxpool2d_optimized_forward(const float* d_input, float* d_output, float* indices,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    maxpool2d_forward_optimized_kernel<<<grid_size, block_size>>>(
    d_input, d_output, indices, batch, h, w, c, pool_size, stride);

    CUDA_CHECK(cudaGetLastError());
}


// Backward
// Fused kernel: Compute grad_bias, grad_weights, and grad_input in one pass
// with loop unrolling for kernel_size=3
__global__ void conv2d_relu_backward_kernel_fused(const float* grad_output, 
                                                   const float* input,
                                                   const float* weights, 
                                                   const float* conv_output, 
                                                   float* grad_input,
                                                   float* grad_weights, 
                                                   float* grad_bias,
                                                   int batch, int in_h, int in_w, int in_c, 
                                                   int out_h, int out_w, int out_c, 
                                                   int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx >= total) return;
    
    int oc = idx % out_c;
    int ow = (idx / out_c) % out_w;
    int oh = (idx / out_c / out_w) % out_h;
    int b = idx / out_c / out_w / out_h;
    
    // Apply ReLU derivative: grad flows back only if conv_output > 0
    float conv_out_val = conv_output[idx];
    float grad_out_val = grad_output[idx];
    
    // Fused ReLU backward: multiply gradient by ReLU mask
    float grad_after_relu = grad_out_val * (conv_out_val > 0.0f ? 1.0f : 0.0f);
    
    // Now this becomes the gradient for Conv2D backward
    atomicAdd(&grad_bias[oc], grad_after_relu);
    
    // Compute Conv2D gradients with the masked gradient
    if (kernel_size == 3) {
        for (int ic = 0; ic < in_c; ++ic) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh * stride - padding + kh;
                if (ih >= 0 && ih < in_h) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow * stride - padding + kw;
                        if (iw >= 0 && iw < in_w) {
                            int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                            int w_idx = oc * in_c * 9 + ic * 9 + kh * 3 + kw;
                            
                            float in_val = input[in_idx];
                            float w_val = weights[w_idx];
                            
                            atomicAdd(&grad_weights[w_idx], in_val * grad_after_relu);
                            atomicAdd(&grad_input[in_idx], w_val * grad_after_relu);
                        }
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih = oh * stride - padding + kh;
                if (ih >= 0 && ih < in_h) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int iw = ow * stride - padding + kw;
                        if (iw >= 0 && iw < in_w) {
                            int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                            int w_idx = oc * in_c * kernel_size * kernel_size + 
                                       ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            
                            atomicAdd(&grad_weights[w_idx], input[in_idx] * grad_after_relu);
                            atomicAdd(&grad_input[in_idx], weights[w_idx] * grad_after_relu);
                        }
                    }
                }
            }
        }
    }
}

void launch_conv2d_relu_backward_fused(const float* d_grad_output, 
                                       const float* d_input,
                                       const float* d_weights, 
                                       const float* d_conv_output,
                                       float* d_grad_input,
                                       float* d_grad_weights, 
                                       float* d_grad_bias,
                                       int batch, int in_h, int in_w, int in_c,
                                       int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_relu_backward_kernel_fused<<<grid_size, block_size>>>(
        d_grad_output, d_input, d_weights, d_conv_output, d_grad_input,
        d_grad_weights, d_grad_bias,
        batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}
