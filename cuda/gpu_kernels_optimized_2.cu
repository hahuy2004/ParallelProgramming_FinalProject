// Tập trung vào Kernel-Level Optimization 
#include "gpu_kernels_optimized_2.h"
#include "gpu_kernels_naive.h"
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
__global__ void conv2d_relu_kernel(
    const float* input, 
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    int oc = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (oc >= C_out || oh >= H_out || ow >= W_out) return;

    float sum = 0.0f;
    
    if (kernel_size == 3){
        for (int ic = 0; ic < C_in; ic++) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
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
    }
    else{
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
    }
    sum += bias[oc];

    float final_output = fmaxf(0.0f, sum);

    output[oc * H_out * W_out + oh * W_out + ow] = final_output;
}

void launch_conv2d_relu_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C_out, (H_out + 15) / 16, (W_out + 15) / 16);
 
    conv2d_relu_kernel<<<gridDim, blockDim>>>(
        input, weight, bias, output,
        C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool_kernel_unroll(
    const float* input,
    float* output,
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

    float max_val = -1e38f;

    int ih0 = ih_start;
    int iw0 = iw_start;
    if (ih0 < H && iw0 < W) {
        float val = input[c * H * W + ih0 * W + iw0];
        max_val = fmaxf(max_val, val);
    }

    int iw1 = iw_start + 1;
    if (ih0 < H && iw1 < W) {
        float val = input[c * H * W + ih0 * W + iw1];
        max_val = fmaxf(max_val, val);
    }

    int ih1 = ih_start + 1;
    if (ih1 < H && iw0 < W) {
        float val = input[c * H * W + ih1 * W + iw0];
        max_val = fmaxf(max_val, val);
    }

    if (ih1 < H && iw1 < W) {
        float val = input[c * H * W + ih1 * W + iw1];
        max_val = fmaxf(max_val, val);
    }

    output[c * H_out * W_out + oh * W_out + ow] = max_val;
}

void launch_maxpool_unroll_forward(
    const float* input,
    float* output,
    int C, int H, int W,
    int pool_size, int stride)
{
    int H_out = (H - pool_size) / stride + 1;
    int W_out = (W - pool_size) / stride + 1;
    
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C, (H_out + 15) / 16, (W_out + 15) / 16);
    
    maxpool_kernel_unroll<<<gridDim, blockDim>>>(input, output, C, H, W, pool_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool_backward_unroll(
    const float* grad_output,
    const float* input,
    float* grad_input,
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


    float max_val = -1e38f;
    int max_idx = -1; 
 
    int ih0 = ih_start;
    int iw0 = iw_start;
    if (ih0 < H && iw0 < W) {
        int current_idx = c * H * W + ih0 * W + iw0;
        float val = input[current_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = current_idx;
        }
    }


    int iw1 = iw_start + 1;
    if (ih0 < H && iw1 < W) {
        int current_idx = c * H * W + ih0 * W + iw1;
        float val = input[current_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = current_idx;
        }
    }


    int ih1 = ih_start + 1;
    if (ih1 < H && iw0 < W) {
        int current_idx = c * H * W + ih1 * W + iw0;
        float val = input[current_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = current_idx;
        }
    }


    if (ih1 < H && iw1 < W) {
        int current_idx = c * H * W + ih1 * W + iw1;
        float val = input[current_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = current_idx;
        }
    }

    float grad = grad_output[c * H_out * W_out + oh * W_out + ow];


    if (max_idx != -1) { 

        atomicAdd(&grad_input[max_idx], grad);
    }
}

void launch_maxpool_unroll_backward(
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
    
    maxpool_backward_unroll<<<gridDim, blockDim>>>(grad_output, input, grad_input, C, H, W, pool_size, stride);
    
    CUDA_CHECK(cudaGetLastError());
}


// Conv2D input gradient kernel with configurable parameters
__global__ void conv2d_input_grad_unroll_kernel(
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
    if(kernel_size==3){
        for (int oc = 0; oc < C_out; oc++) {
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    // For stride > 1, need to check if this input contributes to output
                    int oh_temp = ih + padding - kh;
                    int ow_temp = iw + padding - kw;
                    
                    if (oh_temp % stride == 0 && ow_temp % stride == 0) {
                        int oh = oh_temp / stride;
                        int ow = ow_temp / stride;
                        
                        if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                            float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                            int weight_idx = ((oc * C_in + ic) * 3 + kh) * 3 + kw;
                            sum += grad * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    else{
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
    }
    
    grad_input[ic * H_in * W_in + ih * W_in + iw] = sum;
}

void launch_conv2d_input_grad_unroll(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(C_in, (H_in + 15) / 16, (W_in + 15) / 16);
    
    conv2d_input_grad_unroll_kernel<<<gridDim, blockDim>>>(
        grad_output, weight, grad_input,
        C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding);
    
    CUDA_CHECK(cudaGetLastError());
}
