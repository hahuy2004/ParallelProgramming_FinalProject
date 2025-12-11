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


// Sử dụng shared memory cho convolution ở giai đoạn forward
__global__ void conv2d_forward_shared(const float* input, float* output,
                                     const float* weights, const float* bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding) {
    extern __shared__ float s[]; 
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    int b = blockIdx.z;
    int oc = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int oh0 = blockIdx.x / (out_w / blockDim.x) * blockDim.y;
    int ow0 = blockIdx.x % (out_w / blockDim.x) * blockDim.x;

    int tile_h = blockDim.y + kernel_size - 1;
    int tile_w = blockDim.x + kernel_size - 1;

    // Tải input vào shared memory
    for (int ic = 0; ic < in_c; ++ic) {
        for (int th = ty; th < tile_h; th += blockDim.y) {
            for (int tw = tx; tw < tile_w; tw += blockDim.x) {

                int ih = oh0 * stride - padding + th;
                int iw = ow0 * stride - padding + tw;

                float val = 0.0f;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = b*in_h*in_w*in_c + ih*in_w*in_c + iw*in_c + ic;
                    val = input[in_idx];
                }

                s[ic * tile_h * tile_w + th * tile_w + tw] = val;
            }
        }
    }

    __syncthreads();

    // Tính output pixel
    int oh = oh0 + ty;
    int ow = ow0 + tx;

    if (oh < out_h && ow < out_w) {

        float sum = bias[oc];

        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {

                    int s_idx = ic * tile_h * tile_w + (ty + kh) * tile_w + (tx + kw);

                    int w_idx = oc * in_c * kernel_size * kernel_size + ic * kernel_size * kernel_size +
                                kh * kernel_size + kw;

                    sum += s[s_idx] * weights[w_idx];
                }
            }
        }

        int out_idx = b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
        output[out_idx] = sum;
    }
}

void launch_conv2d_shared_forward(const float* d_input, float* d_output,
                                   const float* d_weights, const float* d_bias,
                                   int batch, int in_h, int in_w, int in_c,
                                   int out_c, int kernel_size, int stride, int padding) {
    // Mặc định block size
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    dim3 block(16, 16);
    
    // Tính toán grid size
    int grid_x = ((out_w + 16 - 1) / 16) * ((out_h + 16 - 1) / 16);
    dim3 grid(grid_x, out_c, batch); 
    
    // Tính kích thước shared memory
    int tile_h = 16 + kernel_size - 1;
    int tile_w = 16 + kernel_size - 1;
    int shmem_bytes = tile_h * tile_w * in_c * sizeof(float);

    conv2d_forward_shared<<<grid, block, shmem_bytes>>>(
        d_input, d_output, d_weights, d_bias,
        batch, in_h, in_w, in_c, out_c, kernel_size, stride, padding);

    
    CUDA_CHECK(cudaGetLastError());
}

// Sử dụng shared memory cho giai đoạn backward
__global__ void conv2d_shared_backward(const float* grad_output, const float* input,
                                        const float* weights, float* grad_input,
                                        float* grad_weights, float* grad_bias,
                                        int batch, int in_h, int in_w, int in_c,
                                        int out_c, int kernel_size, int stride, int padding) {
    
    extern __shared__ float s[];

    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    int b = blockIdx.z;  
    int oc = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int oh0 = blockIdx.x / (out_w / blockDim.x) * blockDim.y;
    int ow0 = blockIdx.x % (out_w / blockDim.x) * blockDim.x;

    int tile_h = blockDim.y + kernel_size - 1;
    int tile_w = blockDim.x + kernel_size - 1;


    for (int ic = 0; ic < in_c; ++ic)
    {
        for (int th = ty; th < tile_h; th += blockDim.y)
        {
            for (int tw = tx; tw < tile_w; tw += blockDim.x)
            {
                int ih = oh0 * stride - padding + th;
                int iw = ow0 * stride - padding + tw;

                float val = 0.0f;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                {
                    int in_idx =
                        b * in_h * in_w * in_c +
                        ih * in_w * in_c +
                        iw * in_c + ic;

                    val = input[in_idx];
                }

                s[ic * tile_h * tile_w + th * tile_w + tw] = val;
            }
        }
    }

    __syncthreads();


    int oh = oh0 + ty;
    int ow = ow0 + tx;

    if (oh >= out_h || ow >= out_w)
        return;

    // grad_output tại vị trí output này
    int go_idx = b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;

    float go_val = grad_output[go_idx];

    // bias
    if (tx == 0 && ty == 0)
        atomicAdd(&grad_bias[oc], go_val);

    for (int ic = 0; ic < in_c; ++ic)
    {
        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                int tile_y = ty + kh;
                int tile_x = tx + kw;

                float v = 0.0f;

                if (tile_y < tile_h && tile_x < tile_w)
                {
                    int s_idx =
                        ic * tile_h * tile_w +
                        tile_y * tile_w + tile_x;

                    v = s[s_idx];
                }

                // gradient weights
                int w_idx = oc * in_c * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;

                atomicAdd(&grad_weights[w_idx], v * go_val);

                // gradient input
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                {
                    int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;

                    atomicAdd(&grad_input[in_idx], weights[w_idx] * go_val);
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
    dim3 block(16, 16);

    int grid_x = ((out_w + 16 - 1) / 16) * ((out_h + 16 - 1) / 16);
    dim3 grid(grid_x, out_c, batch);
    
    int tile_h = 16 + kernel_size - 1;
    int tile_w = 16 + kernel_size - 1;
    int shmem_bytes = tile_h * tile_w * in_c * sizeof(float);

    conv2d_shared_backward<<<grid, block, shmem_bytes>>>(
        d_input, d_grad_output, d_weights,
        d_grad_input, d_grad_weights, d_grad_bias,
        batch, in_h, in_w, in_c,
        out_c, kernel_size, stride, padding
    );

    CUDA_CHECK(cudaGetLastError());
}