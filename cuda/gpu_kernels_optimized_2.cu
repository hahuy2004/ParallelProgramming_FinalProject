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
// Fused Conv2D + ReLU + Bias
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

// Hàm này áp dụng cho pool size = 2 theo đề bài quy định
__global__ void maxpool2d_forward_optimized_kernel(const float* input, float* output,
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
            
            // Tìm max của 4 giá trị (unrolled comparison)
            max_val = fmaxf(fmaxf(val0, val1), fmaxf(val2, val3));
        }
        else{
            for (int ph = 0; ph < pool_size; ++ph) {
                for (int pw = 0; pw < pool_size; ++pw) {
                    int ih = oh * stride + ph;
                    int iw = ow * stride + pw;
                    int in_idx = b * h * w * c + ih * w * c + iw * c + ch;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }

        output[idx] = max_val;
    }
}

void launch_maxpool2d_optimized_forward(const float* d_input, float* d_output,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    maxpool2d_forward_optimized_kernel<<<grid_size, block_size>>>(
    d_input, d_output, batch, h, w, c, pool_size, stride);

    CUDA_CHECK(cudaGetLastError());
}


// Backward
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

__global__ void maxpool2d_backward_optimized_kernel(const float* grad_output, const float* input,
                                        const float* output, float* grad_input,
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

        float grad_out_val = grad_output[idx];
        float max_val = output[idx];

        // Loop unrolling cho pool_size=2 (trường hợp phổ biến nhất)
        if (pool_size == 2 && stride == 2) {
            // Unroll hoàn toàn cho 2x2 pooling với stride=2
            int ih0 = oh * stride;
            int iw0 = ow * stride;
            int base_idx = b * h * w * c + ih0 * w * c + iw0 * c + ch;
            
            // Check 4 vị trí và route gradient (fully unrolled)
            int in_idx0 = base_idx;
            int in_idx1 = base_idx + c;
            int in_idx2 = base_idx + w * c;
            int in_idx3 = base_idx + w * c + c;
            
            // Route gradient đến vị trí có giá trị = max_val (unrolled comparisons)
            if (input[in_idx0] == max_val) {
                atomicAdd(&grad_input[in_idx0], grad_out_val);
            }
            if (input[in_idx1] == max_val) {
                atomicAdd(&grad_input[in_idx1], grad_out_val);
            }
            if (input[in_idx2] == max_val) {
                atomicAdd(&grad_input[in_idx2], grad_out_val);
            }
            if (input[in_idx3] == max_val) {
                atomicAdd(&grad_input[in_idx3], grad_out_val);
            }
        }else {
            for (int ph = 0; ph < pool_size; ++ph) {
                for (int pw = 0; pw < pool_size; ++pw) {
                    int ih = oh * stride + ph;
                    int iw = ow * stride + pw;
                    int in_idx = b * h * w * c + ih * w * c + iw * c + ch;

                    // Route gradient to the max position
                    if (input[in_idx] == max_val) {
                        atomicAdd(&grad_input[in_idx], grad_out_val);
                    }
                }
            }
        }
    }
}

void launch_maxpool2d_optimized_backward(const float* d_grad_output, const float* d_input,
                                const float* d_output, float* d_grad_input,
                                int batch, int h, int w, int c,
                                int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    maxpool2d_backward_optimized_kernel<<<grid_size, block_size>>>(
    d_grad_output, d_input, d_output, d_grad_input,
    batch, h, w, c, pool_size, stride);

    CUDA_CHECK(cudaGetLastError());
}


// Loop unrolling cho trường hợp scale factor ==2
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
        
        // Loop unrolling cho scale_factor=2 
        if (scale_factor == 2) {
            // Unroll hoàn toàn cho scale_factor=2
            // Loại bỏ loop overhead, compiler có thể optimize tốt hơn
            int oh0 = ih * 2;
            int ow0 = iw * 2;
            int base_idx = b * out_h * out_w * c + oh0 * out_w * c + ow0 * c + ch;
            
            // Sum 4 giá trị từ output (fully unrolled)
            grad_sum = grad_output[base_idx] +                    // (oh0, ow0)
                       grad_output[base_idx + c] +                // (oh0, ow0+1)
                       grad_output[base_idx + out_w * c] +        // (oh0+1, ow0)
                       grad_output[base_idx + out_w * c + c];     // (oh0+1, ow0+1)
        } 
        else {
            // Fallback cho scale_factor lớn hơn 4 (hiếm khi dùng)
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
