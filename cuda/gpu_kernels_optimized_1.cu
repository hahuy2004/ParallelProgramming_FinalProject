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

// Constant memory for biases (64KB available, plenty for all biases)
__constant__ float c_conv1_bias[256];
__constant__ float c_conv2_bias[128];
__constant__ float c_conv3_bias[128];
__constant__ float c_conv4_bias[256];
__constant__ float c_conv5_bias[3];

// Optimized Conv2D kernel with Shared Memory Tiling
__global__ void conv2d_kernel_optimized(
    const float* input,   // [C_in, H, W]
    const float* weight,  // [C_out, C_in, K, K]
    const float* bias,    // [C_out]
    float* output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{
    // Thread indexing
    int oc = blockIdx.x;  // Output channel
    int tile_oh = blockIdx.y * blockDim.y;  // Tile start in output height
    int tile_ow = blockIdx.z * blockDim.z;  // Tile start in output width
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Output position for this thread
    int oh = tile_oh + ty;
    int ow = tile_ow + tz;
    
    // Shared memory for input tile
    // Size: [TILE_H + K - 1][TILE_W + K - 1] to include halo region
    const int SMEM_H = TILE_SIZE + kernel_size - 1;
    const int SMEM_W = TILE_SIZE + kernel_size - 1;
    
    extern __shared__ float smem[];  // Dynamic shared memory
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < C_in; ic++) {
        // Cooperatively load input tile into shared memory
        // Each thread loads multiple elements if needed
        int smem_size = SMEM_H * SMEM_W;
        int threads_per_block = blockDim.y * blockDim.z;
        int tid = ty * blockDim.z + tz;
        
        // Calculate input tile boundaries (with stride consideration)
        int tile_ih_start = tile_oh * stride - padding;
        int tile_iw_start = tile_ow * stride - padding;
        
        // Load input tile cooperatively
        for (int idx = tid; idx < smem_size; idx += threads_per_block) {
            int smem_h = idx / SMEM_W;
            int smem_w = idx % SMEM_W;
            
            int ih = tile_ih_start + smem_h;
            int iw = tile_iw_start + smem_w;
            
            float val = 0.0f;
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                val = input[ic * H_in * W_in + ih * W_in + iw];
            }
            smem[smem_h * SMEM_W + smem_w] = val;
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        if (oh < H_out && ow < W_out) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    // Position in shared memory
                    int smem_h = ty * stride + kh;
                    int smem_w = tz * stride + kw;
                    
                    float input_val = smem[smem_h * SMEM_W + smem_w];
                    
                    int weight_idx = ((oc * C_in + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input_val * weight[weight_idx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (oh < H_out && ow < W_out) {
        sum += bias[oc];
        output[oc * H_out * W_out + oh * W_out + ow] = sum;
    }
}

void launch_conv2d_shared_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding)
{   
    // Calculate shared memory size
    int smem_h = TILE_SIZE + kernel_size - 1;
    int smem_w = TILE_SIZE + kernel_size - 1;
    int smem_size = smem_h * smem_w * sizeof(float);
    
    dim3 blockDim(1, TILE_SIZE, TILE_SIZE);
    dim3 gridDim(C_out, (H_out + TILE_SIZE - 1) / TILE_SIZE, (W_out + TILE_SIZE - 1) / TILE_SIZE);
    
    conv2d_kernel_optimized<<<gridDim, blockDim, smem_size>>>(
        input, weight, bias, output,
        C_in, H_in, W_in, C_out, H_out, W_out, 
        kernel_size, stride, padding);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Copy bias to constant memory for faster access
void copy_bias_to_constant_memory(
    const float* d_bias, int size, int layer)
{
    switch(layer) {
        case 0:  // Conv1: 256
            CUDA_CHECK(cudaMemcpyToSymbol(c_conv1_bias, d_bias, size * sizeof(float)));
            break;
        case 1:  // Conv2: 128
            CUDA_CHECK(cudaMemcpyToSymbol(c_conv2_bias, d_bias, size * sizeof(float)));
            break;
        case 2:  // Conv3: 128
            CUDA_CHECK(cudaMemcpyToSymbol(c_conv3_bias, d_bias, size * sizeof(float)));
            break;
        case 3:  // Conv4: 256
            CUDA_CHECK(cudaMemcpyToSymbol(c_conv4_bias, d_bias, size * sizeof(float)));
            break;
        case 4:  // Conv5: 3
            CUDA_CHECK(cudaMemcpyToSymbol(c_conv5_bias, d_bias, size * sizeof(float)));
            break;
        default:
            fprintf(stderr, "Invalid layer index: %d\n", layer);
    }
}

