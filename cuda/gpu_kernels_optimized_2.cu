// Tập trung vào Kernel-Level Optimization 
#include "gpu_kernels_optimized_2.h"
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
// Fused Cov2d + Relu + Bias 
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

    // Fused ReLU activation: max(0, sum)
    output[idx] = fmaxf(0.0f, sum);
}

void launch_conv2d_forward_relu_fused2(const float* d_input, float* d_output,
                                      const float* d_weights, const float* d_bias,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding, cudaStream_t stream) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_forward_relu_fused<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

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

void launch_conv2d_forward2(const float* d_input, float* d_output,
                           const float* d_weights, const float* d_bias,
                           int batch, int in_h, int in_w, int in_c,
                           int out_c, int kernel_size, int stride, int padding, cudaStream_t stream) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_forward_kernel<<<grid_size, block_size,0, stream>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);
    
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

void launch_maxpool2d_optimized_forward2(const float* d_input, float* d_output, float* indices,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride, cudaStream_t stream) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    maxpool2d_forward_optimized_kernel<<<grid_size, block_size, 0, stream>>>(
    d_input, d_output, indices, batch, h, w, c, pool_size, stride);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_forward_kernel(const float* input, float* output,
                                          int batch, int in_h, int in_w, int c, int out_h, int out_w,
                                          int scale_factor) {
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

void launch_upsample2d_forward2(const float* d_input, float* d_output,
                               int batch, int in_h, int in_w, int c,
                               int scale_factor, cudaStream_t stream) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    int total = batch * out_h * out_w * c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    upsample2d_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, batch, in_h, in_w, c, out_h, out_w, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}


//=====================================================================================
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
    
    // Fused ReLU backward: multiply gradient by ReLU mask
    float grad_after_relu = grad_output[idx] * (conv_output[idx] > 0.0f ? 1.0f : 0.0f);
    
    // Now this becomes the gradient for Conv2D backward
    atomicAdd(&grad_bias[oc], grad_after_relu);
    
    // Compute Conv2D gradients with the masked gradient
    for (int ic = 0; ic < in_c; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih >= 0 && ih < in_h) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
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
}

void launch_conv2d_relu_backward_fused2(const float* d_grad_output, 
                                       const float* d_input,
                                       const float* d_weights, 
                                       const float* d_conv_output,
                                       float* d_grad_input,
                                       float* d_grad_weights, 
                                       float* d_grad_bias,
                                       int batch, int in_h, int in_w, int in_c,
                                       int out_c, int kernel_size, int stride, int padding, cudaStream_t stream) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_relu_backward_kernel_fused<<<grid_size, block_size, 0, stream>>>(
        d_grad_output, d_input, d_weights, d_conv_output, d_grad_input,
        d_grad_weights, d_grad_bias,
        batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}


__global__ void conv2d_backward_kernel(const float* grad_output, const float* input,
                                       const float* weights, float* grad_input,
                                       float* grad_weights, float* grad_bias,
                                       int batch, int in_h, int in_w, int in_c, int out_h, int out_w,
                                       int out_c, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_h * out_w * out_c;
    
    if (idx < total) {
        int oc = idx % out_c;
        int ow = (idx / out_c) % out_w;
        int oh = (idx / out_c / out_w) % out_h;
        int b = idx / out_c / out_w / out_h;
        
        float grad_out_val = grad_output[idx];
        
        // Accumulate gradient for bias
        atomicAdd(&grad_bias[oc], grad_out_val);
        
        // Compute gradients for weights and input
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                        int w_idx = oc * in_c * kernel_size * kernel_size + 
                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        // Gradient w.r.t. weights: grad_w = input * grad_output
                        atomicAdd(&grad_weights[w_idx], input[in_idx] * grad_out_val);
                        
                        // Gradient w.r.t. input: grad_input = weights * grad_output
                        atomicAdd(&grad_input[in_idx], weights[w_idx] * grad_out_val);
                    }
                }
            }
        }
    }
}

void launch_conv2d_backward2(const float* d_grad_output, const float* d_input,
                            const float* d_weights, float* d_grad_input,
                            float* d_grad_weights, float* d_grad_bias,
                            int batch, int in_h, int in_w, int in_c,
                            int out_c, int kernel_size, int stride, int padding, cudaStream_t stream) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    int total = batch * out_h * out_w * out_c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_grad_output, d_input, d_weights, d_grad_input,
        d_grad_weights, d_grad_bias,
        batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}

// MaxPool2D Backward: route gradient to the max location
__global__ void maxpool2d_backward_kernel(const float* grad_output, const float* indices,
                                          float* grad_input, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int max_idx = (int)indices[i];
        if (max_idx != -1) atomicAdd(&grad_input[max_idx], grad_output[i]);
    }
}

void launch_maxpool2d_backward2(const float* d_grad_output, float* d_input, const float* indices,
                               const float* d_output, float* d_grad_input,
                               int batch, int h, int w, int c,
                               int pool_size, int stride, cudaStream_t stream) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    int total = batch * out_h * out_w * c;
    
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    maxpool2d_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_grad_output, indices, d_grad_input, total);
    
    CUDA_CHECK(cudaGetLastError());
}


// UpSample2D Backward
__global__ void upsample2d_backward_kernel(const float* grad_output, float* grad_input,
                                           int batch, int in_h, int in_w, int c, int out_h, int out_w,
                                           int scale_factor) {  

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch * out_h * out_w * c;  // số phần tử của grad_output

    if (idx < total_out) {
        int ch = idx % c;
        int ow = (idx / c) % out_w;
        int oh = (idx / c / out_w) % out_h;
        int b  = idx / c / out_w / out_h;

        // ánh xạ về tọa độ input gốc
        int in_h_idx = oh / scale_factor;
        int in_w_idx = ow / scale_factor;

        int in_idx = b * in_h * in_w * c + in_h_idx * in_w * c + in_w_idx * c + ch;
        atomicAdd(&grad_input[in_idx], grad_output[idx]);
    }
}

void launch_upsample2d_backward2(const float* d_grad_output, float* d_grad_input,
                                int batch, int in_h, int in_w, int c,
                                int scale_factor, cudaStream_t stream) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    int total_out = batch * out_h * out_w * c;
    
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    
    upsample2d_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_grad_output, d_grad_input, batch, in_h, in_w, c, out_h, out_w, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

//=====================================================
// Untility
// Forward
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

void launch_mse_loss2(const float* d_input, const float* d_output,
                     float* d_loss, int size, cudaStream_t stream) {
    // Initialize loss to 0
    CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float), stream));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    mse_loss_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_output, d_loss, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void zero_grad_kernel(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// Backward

void launch_zero_grad2(float* d_grad, int size, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    zero_grad_kernel<<<grid_size, block_size, 0, stream>>>(d_grad, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sgd_update_kernel(float* weights, const float* grad,
                                  float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad[idx];
    }
}

void launch_sgd_update2(float* d_weights, const float* d_grad,
                       float learning_rate, int size, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sgd_update_kernel<<<grid_size, block_size, 0, stream>>>(d_weights, d_grad, learning_rate, size);
    CUDA_CHECK(cudaGetLastError());
}

// MSE Loss Backward: compute gradient of loss w.r.t. output
__global__ void mse_loss_backward_kernel(const float* output, const float* target,
                                         float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = 2.0f * (output[idx] - target[idx]) / size;
    }
}

void launch_mse_loss_backward2(const float* d_output, const float* d_target,
                              float* d_grad_output, int size, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    mse_loss_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_output, d_target, d_grad_output, size);
    CUDA_CHECK(cudaGetLastError());
}

