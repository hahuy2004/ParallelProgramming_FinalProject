#include "../include/autoencoder_gpu_optimized_2.h"
#include "../cuda/gpu_kernels_optimized_2.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void AutoencoderGPUOptimized2::setup_pipeline_resources(int batch_size) {
    if (batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float) == pinned_buffer_size_) return;

    // 1. Giải phóng nếu đã tồn tại
    if (pinned_buffer_size_ > 0) free_pipeline_resources();

    // 2. Tạo Streams
    CUDA_CHECK(cudaStreamCreate(&stream_h2d_[0]));
    CUDA_CHECK(cudaStreamCreate(&stream_h2d_[1]));
    CUDA_CHECK(cudaStreamCreate(&stream_comp_));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h_));

    // 3. Tạo Events
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_complete_event_[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_complete_event_[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&comp_complete_event_[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&comp_complete_event_[1], cudaEventDisableTiming));

    // 4. Cấp phát Pinned Memory
    pinned_buffer_size_ = (size_t)batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float);
    
    CUDA_CHECK(cudaHostAlloc((void**)&h_input_pinned_[0], pinned_buffer_size_, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_input_pinned_[1], pinned_buffer_size_, cudaHostAllocMapped));
    
    CUDA_CHECK(cudaHostAlloc((void**)&h_loss_pinned_[0], sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_loss_pinned_[1], sizeof(float), cudaHostAllocMapped));
    
    CUDA_CHECK(cudaMalloc(&d_input_next_, pinned_buffer_size_));
    CUDA_CHECK(cudaMalloc(&d_loss_next_, sizeof(float)));

}

void AutoencoderGPUOptimized2::free_pipeline_resources() {
    if (pinned_buffer_size_ == 0) return;

    cudaStreamDestroy(stream_h2d_[0]);
    cudaStreamDestroy(stream_h2d_[1]);
    cudaStreamDestroy(stream_comp_);
    cudaStreamDestroy(stream_d2h_);

    cudaEventDestroy(h2d_complete_event_[0]);
    cudaEventDestroy(h2d_complete_event_[1]);
    cudaEventDestroy(comp_complete_event_[0]);
    cudaEventDestroy(comp_complete_event_[1]);

    cudaFreeHost(h_input_pinned_[0]);
    cudaFreeHost(h_input_pinned_[1]);
    cudaFreeHost(h_loss_pinned_[0]);
    cudaFreeHost(h_loss_pinned_[1]);
    
    if (d_input_next_) cudaFree(d_input_next_);
    if (d_loss_next_) cudaFree(d_loss_next_);

    pinned_buffer_size_ = 0;
}

AutoencoderGPUOptimized2::AutoencoderGPUOptimized2() : current_batch_size_(0) {
    // Initialize all device pointers to nullptr
    d_conv1_weights_ = nullptr;
    d_conv1_bias_ = nullptr;
    d_conv2_weights_ = nullptr;
    d_conv2_bias_ = nullptr;
    d_conv3_weights_ = nullptr;
    d_conv3_bias_ = nullptr;
    d_conv4_weights_ = nullptr;
    d_conv4_bias_ = nullptr;
    d_conv5_weights_ = nullptr;
    d_conv5_bias_ = nullptr;
    
    d_input_ = nullptr;
    d_conv1_out_ = nullptr;
    d_pool1_out_ = nullptr;
    d_indices1_ = nullptr;
    d_conv2_out_ = nullptr;
    d_pool2_out_ = nullptr;
    d_indices2_ = nullptr;
    d_conv3_out_ = nullptr;
    d_up1_out_ = nullptr;
    d_conv4_out_ = nullptr;
    d_up2_out_ = nullptr;
    d_conv5_out_ = nullptr;
    
    d_grad_conv5_out_ = nullptr;
    d_grad_up2_out_ = nullptr;
    d_grad_conv4_out_ = nullptr;
    d_grad_up1_out_ = nullptr;
    d_grad_conv3_out_ = nullptr;
    d_grad_pool2_out_ = nullptr;
    d_grad_conv2_out_ = nullptr;
    d_grad_pool1_out_ = nullptr;
    d_grad_conv1_out_ = nullptr;
    
    d_grad_conv1_weights_ = nullptr;
    d_grad_conv1_bias_ = nullptr;
    d_grad_conv2_weights_ = nullptr;
    d_grad_conv2_bias_ = nullptr;
    d_grad_conv3_weights_ = nullptr;
    d_grad_conv3_bias_ = nullptr;
    d_grad_conv4_weights_ = nullptr;
    d_grad_conv4_bias_ = nullptr;
    d_grad_conv5_weights_ = nullptr;
    d_grad_conv5_bias_ = nullptr;
    
    d_loss_ = nullptr;
    
    initialize_weights();
}

AutoencoderGPUOptimized2::~AutoencoderGPUOptimized2() {
    free_device_memory();
    free_pipeline_resources();
}

void AutoencoderGPUOptimized2::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initilization
    auto init_and_upload = [&](float** d_ptr, int size, int fan_in) {
        std::vector<float> h_weights(size);
        float std = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        for (auto& w : h_weights) w = dist(gen);
        
        CUDA_CHECK(cudaMalloc(d_ptr, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*d_ptr, h_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    };
    
    auto init_bias = [&](float** d_ptr, int size) {
        CUDA_CHECK(cudaMalloc(d_ptr, size * sizeof(float)));
        CUDA_CHECK(cudaMemset(*d_ptr, 0, size * sizeof(float)));
    };
    
    // Initialize weights
    init_and_upload(&d_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3, INPUT_C * 3 * 3);
    init_bias(&d_conv1_bias_, CONV1_FILTERS);
    
    init_and_upload(&d_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv2_bias_, CONV2_FILTERS);
    
    init_and_upload(&d_conv3_weights_, LATENT_C * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv3_bias_, LATENT_C);
    
    init_and_upload(&d_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv4_bias_, CONV1_FILTERS);
    
    init_and_upload(&d_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv5_bias_, INPUT_C);
    
    CUDA_CHECK(cudaMalloc(&d_loss_, sizeof(float)));
    
    // Allocate gradient buffers for weights
    CUDA_CHECK(cudaMalloc(&d_grad_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv1_bias_, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv2_bias_, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv3_weights_, LATENT_C * LATENT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv3_bias_, LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4_bias_, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv5_bias_, INPUT_C * sizeof(float)));
    
    std::cout << "GPU weights initialized" << std::endl;
}

void AutoencoderGPUOptimized2::allocate_device_memory(int batch_size) {
    if (batch_size == current_batch_size_) return;

    setup_pipeline_resources(batch_size);
    
    // Free old memory
    if (current_batch_size_ > 0) {
        cudaFree(d_input_);
        cudaFree(d_conv1_out_);
        cudaFree(d_pool1_out_);
        cudaFree(d_indices1_);
        cudaFree(d_conv2_out_);
        cudaFree(d_pool2_out_);
        cudaFree(d_indices2_);
        cudaFree(d_conv3_out_);
        cudaFree(d_up1_out_);
        cudaFree(d_conv4_out_);
        cudaFree(d_up2_out_);
        cudaFree(d_conv5_out_);
        
        cudaFree(d_grad_conv5_out_);
        cudaFree(d_grad_up2_out_);
        cudaFree(d_grad_conv4_out_);
        cudaFree(d_grad_up1_out_);
        cudaFree(d_grad_conv3_out_);
        cudaFree(d_grad_pool2_out_);
        cudaFree(d_grad_conv2_out_);
        cudaFree(d_grad_pool1_out_);
        cudaFree(d_grad_conv1_out_);
    }
    
    current_batch_size_ = batch_size;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input_, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices1_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out_, batch_size * 16 * 16 * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices2_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_out_, batch_size * 16 * 16 * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_out_, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    
    // Allocate gradient buffers for activations
    CUDA_CHECK(cudaMalloc(&d_grad_conv5_out_, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up2_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up1_out_, batch_size * 16 * 16 * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv3_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool2_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv2_out_, batch_size * 16 * 16 * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool1_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv1_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
}

void AutoencoderGPUOptimized2::free_device_memory() {
    if (d_conv1_weights_) cudaFree(d_conv1_weights_);
    if (d_conv1_bias_) cudaFree(d_conv1_bias_);
    if (d_conv2_weights_) cudaFree(d_conv2_weights_);
    if (d_conv2_bias_) cudaFree(d_conv2_bias_);
    if (d_conv3_weights_) cudaFree(d_conv3_weights_);
    if (d_conv3_bias_) cudaFree(d_conv3_bias_);
    if (d_conv4_weights_) cudaFree(d_conv4_weights_);
    if (d_conv4_bias_) cudaFree(d_conv4_bias_);
    if (d_conv5_weights_) cudaFree(d_conv5_weights_);
    if (d_conv5_bias_) cudaFree(d_conv5_bias_);
    
    if (d_input_) cudaFree(d_input_);
    if (d_conv1_out_) cudaFree(d_conv1_out_);
    if (d_pool1_out_) cudaFree(d_pool1_out_);
    if (d_indices1_) cudaFree(d_indices1_);
    if (d_conv2_out_) cudaFree(d_conv2_out_);
    if (d_pool2_out_) cudaFree(d_pool2_out_);
    if (d_indices2_) cudaFree(d_indices2_);
    if (d_conv3_out_) cudaFree(d_conv3_out_);
    if (d_up1_out_) cudaFree(d_up1_out_);
    if (d_conv4_out_) cudaFree(d_conv4_out_);
    if (d_up2_out_) cudaFree(d_up2_out_);
    if (d_conv5_out_) cudaFree(d_conv5_out_);
    
    if (d_grad_conv5_out_) cudaFree(d_grad_conv5_out_);
    if (d_grad_up2_out_) cudaFree(d_grad_up2_out_);
    if (d_grad_conv4_out_) cudaFree(d_grad_conv4_out_);
    if (d_grad_up1_out_) cudaFree(d_grad_up1_out_);
    if (d_grad_conv3_out_) cudaFree(d_grad_conv3_out_);
    if (d_grad_pool2_out_) cudaFree(d_grad_pool2_out_);
    if (d_grad_conv2_out_) cudaFree(d_grad_conv2_out_);
    if (d_grad_pool1_out_) cudaFree(d_grad_pool1_out_);
    if (d_grad_conv1_out_) cudaFree(d_grad_conv1_out_);
    
    if (d_grad_conv1_weights_) cudaFree(d_grad_conv1_weights_);
    if (d_grad_conv1_bias_) cudaFree(d_grad_conv1_bias_);
    if (d_grad_conv2_weights_) cudaFree(d_grad_conv2_weights_);
    if (d_grad_conv2_bias_) cudaFree(d_grad_conv2_bias_);
    if (d_grad_conv3_weights_) cudaFree(d_grad_conv3_weights_);
    if (d_grad_conv3_bias_) cudaFree(d_grad_conv3_bias_);
    if (d_grad_conv4_weights_) cudaFree(d_grad_conv4_weights_);
    if (d_grad_conv4_bias_) cudaFree(d_grad_conv4_bias_);
    if (d_grad_conv5_weights_) cudaFree(d_grad_conv5_weights_);
    if (d_grad_conv5_bias_) cudaFree(d_grad_conv5_bias_);
    
    if (d_loss_) cudaFree(d_loss_);
}

void AutoencoderGPUOptimized2::forward_gpu_optimized(int batch_size, cudastream_t stream) {
    // Encoder with FUSED operations
    // Fused Conv2D + ReLU + Bias
    launch_conv2d_forward_relu_fused(d_input_, d_conv1_out_, 
                                    d_conv1_weights_, d_conv1_bias_,
                                    batch_size, INPUT_H, INPUT_W, INPUT_C, 
                                    CONV1_FILTERS, 3, 1, 1, stream);
    
    // MaxPool loop unrolling 
    launch_maxpool2d_optimized_forward(d_conv1_out_, d_pool1_out_, d_indices1_,
                                    batch_size, INPUT_H, INPUT_W, CONV1_FILTERS,
                                    2, 2, stream);
    
    // Fused Conv2D + ReLU + Bias
    launch_conv2d_forward_relu_fused(d_pool1_out_, d_conv2_out_, 
                                d_conv2_weights_, d_conv2_bias_,
                                batch_size, 16, 16, 
                                CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1, stream);
    
    // MaxPool: (16, 16, 128) -> (8, 8, 128)
    launch_maxpool2d_optimized_forward(d_conv2_out_, d_pool2_out_, d_indices2_,
                                batch_size, 16, 16, CONV2_FILTERS, 2, 2, stream);
    
    // Decoder
    // Conv3 + ReLU (fused): (8, 8, 128) -> (8, 8, 128)
    launch_conv2d_forward_relu_fused(d_pool2_out_, d_conv3_out_, d_conv3_weights_, d_conv3_bias_,
                                batch_size, LATENT_H, LATENT_W, LATENT_C, LATENT_C, 3, 1, 1, stream);
    
    // Upsample
    launch_upsample2d_forward(d_conv3_out_, d_up1_out_,
                             batch_size, LATENT_H, LATENT_W, LATENT_C, 2, stream);
    
    // Fused Conv2D + ReLU + Bias
    launch_conv2d_forward_relu_fused(d_up1_out_, d_conv4_out_, d_conv4_weights_, d_conv4_bias_,
                               batch_size, 16, 16, LATENT_C, CONV1_FILTERS, 3, 1, 1, stream);
    
    // Upsample: (16, 16, 256) -> (32, 32, 256)
    launch_upsample2d_forward(d_conv4_out_, d_up2_out_,
                             batch_size, 16, 16, CONV1_FILTERS, 2, stream);
    
    // Conv5 (no activation): (32, 32, 256) -> (32, 32, 3)
    launch_conv2d_forward(d_up2_out_, d_conv5_out_, d_conv5_weights_, d_conv5_bias_,
                         batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_C, 3, 1, 1, stream);
}

float AutoencoderGPUOptimized2::compute_loss_gpu(int batch_size, cudaStream_t stream) {
    int size = batch_size * INPUT_H * INPUT_W * INPUT_C;
    launch_mse_loss(d_input_, d_conv5_out_, d_loss_next_, size, stream);
    return 0.0f;
}

void AutoencoderGPUOptimized2::backward_gpu_optimized(int batch_size, cudaStream_t stream) {
    int size = batch_size * INPUT_H * INPUT_W * INPUT_C;
    
    // Zero out all gradients for weights and biases
    launch_zero_grad(d_grad_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3, stream);
    launch_zero_grad(d_grad_conv1_bias_, CONV1_FILTERS, stream);
    launch_zero_grad(d_grad_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3, stream);
    launch_zero_grad(d_grad_conv2_bias_, CONV2_FILTERS, stream);
    launch_zero_grad(d_grad_conv3_weights_, LATENT_C * LATENT_C * 3 * 3, stream);
    launch_zero_grad(d_grad_conv3_bias_, LATENT_C, stream);
    launch_zero_grad(d_grad_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3, stream);
    launch_zero_grad(d_grad_conv4_bias_, CONV1_FILTERS, stream);
    launch_zero_grad(d_grad_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3, stream);
    launch_zero_grad(d_grad_conv5_bias_, INPUT_C, stream);
    
    // Zero out activation gradients
    launch_zero_grad(d_grad_conv5_out_, batch_size * INPUT_H * INPUT_W * INPUT_C, stream);
    launch_zero_grad(d_grad_up2_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS, stream);
    launch_zero_grad(d_grad_conv4_out_, batch_size * 16 * 16 * CONV1_FILTERS, stream);
    launch_zero_grad(d_grad_up1_out_, batch_size * 16 * 16 * LATENT_C, stream);
    launch_zero_grad(d_grad_conv3_out_, batch_size * LATENT_H * LATENT_W * LATENT_C, stream);
    launch_zero_grad(d_grad_pool2_out_, batch_size * LATENT_H * LATENT_W * LATENT_C, stream);
    launch_zero_grad(d_grad_conv2_out_, batch_size * 16 * 16 * CONV2_FILTERS, stream);
    launch_zero_grad(d_grad_pool1_out_, batch_size * 16 * 16 * CONV1_FILTERS, stream);
    launch_zero_grad(d_grad_conv1_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS, stream);
    
    // Compute gradient of loss w.r.t. output
    launch_mse_loss_backward(d_conv5_out_, d_input_, d_grad_conv5_out_, size, stream);
    
    // Backward through decoder
    // Conv5 backward: (32, 32, 3) <- (32, 32, 256)
    launch_conv2d_backward(d_grad_conv5_out_, d_up2_out_, d_conv5_weights_, 
                          d_grad_up2_out_, d_grad_conv5_weights_, d_grad_conv5_bias_,
                          batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_C, 3, 1, 1, stream);
    
    // Upsample2 backward: (16, 16, 256) <- (32, 32, 256)
    launch_upsample2d_backward(d_grad_up2_out_, d_grad_conv4_out_,
                              batch_size, 16, 16, CONV1_FILTERS, 2, stream);
    
    // FUSED: Conv4 + ReLU4 backward: (16, 16, 128) <- (16, 16, 256)
    // Replaces: ReLU4_backward + Conv4_backward (2 kernels → 1 kernel)
    launch_conv2d_relu_backward_fused(
        d_grad_conv4_out_,          
        d_up1_out_,                 
        d_conv4_weights_,         
        d_conv4_out_,               
        d_grad_up1_out_,           
        d_grad_conv4_weights_,     
        d_grad_conv4_bias_,     
        batch_size, 16, 16, LATENT_C, CONV1_FILTERS, 3, 1, 1, stream
    );
    
    // Upsample1 backward: (8, 8, 128) <- (16, 16, 128)
    launch_upsample2d_backward(d_grad_up1_out_, d_grad_conv3_out_,
                              batch_size, LATENT_H, LATENT_W, LATENT_C, 2, stream);
    
    // FUSED: Conv3 + ReLU3 backward: (8, 8, 128) <- (8, 8, 128)
    launch_conv2d_relu_backward_fused(
        d_grad_conv3_out_,           
        d_pool2_out_,             
        d_conv3_weights_,           
        d_conv3_out_,          
        d_grad_pool2_out_,         
        d_grad_conv3_weights_,    
        d_grad_conv3_bias_,        
        batch_size, LATENT_H, LATENT_W, LATENT_C, LATENT_C, 3, 1, 1, stream
    );
    
    // ENCODER BACKWARD
    
    // MaxPool2 backward: (16, 16, 128) <- (8, 8, 128)
    launch_maxpool2d_backward(d_grad_pool2_out_, d_conv2_out_, d_indices2_, 
                             d_pool2_out_, d_grad_conv2_out_, 
                             batch_size, 16, 16, CONV2_FILTERS, 2, 2, stream);
    
    // FUSED: Conv2 + ReLU2 backward: (16, 16, 256) <- (16, 16, 128)
    launch_conv2d_relu_backward_fused(
        d_grad_conv2_out_,           
        d_pool1_out_,                
        d_conv2_weights_,            
        d_conv2_out_,                
        d_grad_pool1_out_,           
        d_grad_conv2_weights_,       
        d_grad_conv2_bias_,          
        batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1, stream
    );
    
    // MaxPool1 backward: (32, 32, 256) <- (16, 16, 256)
    launch_maxpool2d_backward(d_grad_pool1_out_, d_conv1_out_, d_indices1_, 
                            d_pool1_out_, d_grad_conv1_out_, 
                            batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2, stream);
    
    // FUSED: Conv1 + ReLU1 backward: (32, 32, 3) <- (32, 32, 256)
    float* d_grad_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grad_input, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    launch_zero_grad(d_grad_input, batch_size * INPUT_H * INPUT_W * INPUT_C, stream);
    
    launch_conv2d_relu_backward_fused(
        d_grad_conv1_out_,          
        d_input_,              
        d_conv1_weights_,           
        d_conv1_out_,               
        d_grad_input,            
        d_grad_conv1_weights_,    
        d_grad_conv1_bias_,        
        batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1, stream
    );
    
    CUDA_CHECK(cudaFree(d_grad_input));
}

void AutoencoderGPUOptimized2::update_weights_gpu(float learning_rate, int batch_size, cudaStream_t stream) {
    // Update all weights using SGD
    launch_sgd_update(d_conv1_weights_, d_grad_conv1_weights_, learning_rate,
                     CONV1_FILTERS * INPUT_C * 3 * 3, stream);
    launch_sgd_update(d_conv1_bias_, d_grad_conv1_bias_, learning_rate, CONV1_FILTERS, stream);
    
    launch_sgd_update(d_conv2_weights_, d_grad_conv2_weights_, learning_rate,
                     CONV2_FILTERS * CONV1_FILTERS * 3 * 3, stream);
    launch_sgd_update(d_conv2_bias_, d_grad_conv2_bias_, learning_rate, CONV2_FILTERS, stream);
    
    launch_sgd_update(d_conv3_weights_, d_grad_conv3_weights_, learning_rate,
                     LATENT_C * LATENT_C * 3 * 3, stream);
    launch_sgd_update(d_conv3_bias_, d_grad_conv3_bias_, learning_rate, LATENT_C, stream);
    
    launch_sgd_update(d_conv4_weights_, d_grad_conv4_weights_, learning_rate,
                     CONV1_FILTERS * LATENT_C * 3 * 3, stream);
    launch_sgd_update(d_conv4_bias_, d_grad_conv4_bias_, learning_rate, CONV1_FILTERS, stream);
    
    launch_sgd_update(d_conv5_weights_, d_grad_conv5_weights_, learning_rate,
                     INPUT_C * CONV1_FILTERS * 3 * 3, stream);
    launch_sgd_update(d_conv5_bias_, d_grad_conv5_bias_, learning_rate, INPUT_C, stream);
}

void AutoencoderGPUOptimized2::train(const std::vector<float>& train_images,
                                     int num_images,
                                     int batch_size,
                                     int epochs,
                                     float learning_rate) {
    std::cout << "Training GPU Autoencoder (Kernel-Level Optimization)" << std::endl;
    std::cout << "Images: " << num_images << ", Batch size: " << batch_size 
              << ", Epochs: " << epochs << ", LR: " << learning_rate << std::endl;
    
    allocate_device_memory(batch_size);
    
    const size_t batch_data_size = (size_t)batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float);
    int num_batches = (num_images + batch_size - 1) / batch_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        cudaEvent_t epoch_start, epoch_stop;
        cudaEventCreate(&epoch_start);
        cudaEventCreate(&epoch_stop);
        cudaEventRecord(epoch_start);
        
        float epoch_loss = 0.0f;

        int log_interval = (num_batches > 100) ? 1 : std::max(1, num_batches / 10);
        
        std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "]" << std::endl;

        float* d_current_input = d_input_;
        float* d_next_input = d_input_next_;
        float* d_current_loss = d_loss_;
        float* d_next_loss = d_loss_next_;
        
        float h_prev_loss = 0.0f; // Loss của batch K-1

        int current_batch_size_gpu = batch_size; // Kích thước batch đang được tính toán trên GPU
        int prev_batch_size_gpu = batch_size;    // Kích thước batch đã được tính toán (K-1)

        int actual_batch_size_0 = std::min(batch_size, num_images);
        size_t actual_data_size_0 = (size_t)actual_batch_size_0 * INPUT_H * INPUT_W * INPUT_C * sizeof(float);
        
        const float* batch_data_0 = &train_images[0];
        
        // Copy Host-to-Pinned Memory (CPU work)
        std::memcpy(h_input_pinned_[0], batch_data_0, actual_data_size_0);
        
        // Chuyển Pinnded-to-Device phi đồng bộ (Stream H2D[0])
        CUDA_CHECK(cudaMemcpyAsync(d_current_input, h_input_pinned_[0], 
                                    actual_data_size_0, 
                                    cudaMemcpyHostToDevice, 
                                    stream_h2d_[0]));
        
        // Ghi lại event H2D hoàn thành
        cudaEventRecord(h2d_complete_event_[0], stream_h2d_[0]);

        
        for (int batch = 0; batch < num_batches; ++batch) {

            int current_idx = batch % 2; 
            next_idx = (batch + 1) % 2;

            if (batch + 1 < num_batches) {
                int start_idx_next = (batch + 1) * batch_size;
                int actual_batch_size_next = std::min(batch_size, num_images - start_idx_next);
                size_t actual_data_size_next = (size_t)actual_batch_size_next * INPUT_H * INPUT_W * INPUT_C * sizeof(float);
                
                const float* batch_data_next = &train_images[start_idx_next * INPUT_H * INPUT_W * INPUT_C];
                
                // Copy Host-to-Pinned Memory (CPU work - Bận rộn)
                std::memcpy(h_input_pinned_[next_idx], batch_data_next, actual_data_size_next);
                
                // Chuyển Pinnded-to-Device phi đồng bộ (Stream H2D[next_idx])
                CUDA_CHECK(cudaMemcpyAsync(d_next_input, h_input_pinned_[next_idx], 
                                            actual_data_size_next, // <-- Dùng kích thước chính xác
                                            cudaMemcpyHostToDevice, 
                                            stream_h2d_[next_idx]));
                                            
                // Ghi lại event khi H2D hoàn thành cho batch K+1
                cudaEventRecord(h2d_complete_event_[next_idx], stream_h2d_[next_idx]);
            }
            current_batch_size_gpu = (batch == 0) 
                ? actual_batch_size_0 // Batch 0: Kích thước đã tính toán ở Khởi tạo
                : std::min(batch_size, num_images - batch * batch_size);

            CUDA_CHECK(cudaStreamWaitEvent(stream_comp_, h2d_complete_event_[current_idx], 0));

            forward_gpu_optimized(batch_size, stream_comp_);
            compute_loss_gpu(batch_size, stream_comp_); // Tính loss lên d_next_loss
            backward_gpu_optimized(batch_size, stream_comp_);
            update_weights_gpu(learning_rate, batch_size, stream_comp_);

            cudaEventRecord(comp_complete_event_[current_idx], stream_comp_);

            if (batch > 0) {
                int prev_idx = (current_idx == 0) ? 1 : 0; 
                
                // D2H (loss) chờ Computation hoàn thành (cho batch K-1)
                CUDA_CHECK(cudaStreamWaitEvent(stream_d2h_, comp_complete_event_[prev_idx], 0));
                
                // Copy Loss của batch K-1 về Host (dùng stream_d2h_)
                // Lưu ý: d_next_loss là kết quả tính toán của Batch K-1 sau khi swap
                CUDA_CHECK(cudaMemcpyAsync(h_loss_pinned_[prev_idx], d_next_loss, 
                                            sizeof(float), cudaMemcpyDeviceToHost, stream_d2h_));
                
                // Đồng bộ hóa D2H: Điểm đồng bộ hóa NHỎ NHẤT (chỉ chờ loss về)
                CUDA_CHECK(cudaStreamSynchronize(stream_d2h_));

                h_prev_loss = *h_loss_pinned_[prev_idx];
                float loss = h_prev_loss / (batch_size * INPUT_H * INPUT_W * INPUT_C);
                epoch_loss += loss;

                if ((batch + 1) % log_interval == 0 || batch == num_batches - 1) {
                    float avg_loss = epoch_loss / (batch + 1);
                    float progress = 100.0f * (batch + 1) / num_batches;
                    std::cout << "  Batch [" << (batch + 1) << "/" << num_batches << "] "
                            << "Avg Loss: " << avg_loss << std::endl;
                }
            }

            // Cập nhật kích thước batch đã hoàn thành để dùng cho lần tính toán Loss tiếp theo
            prev_batch_size_gpu = current_batch_size_gpu;

            std::swap(d_current_input, d_next_input);
            std::swap(d_current_loss, d_next_loss);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_comp_));
        
        // 2. Lấy Loss của Batch N-1 về Host (dùng d_current_loss sau khi swap)
        int final_idx = (num_batches - 1) % 2;
        CUDA_CHECK(cudaMemcpy(h_loss_pinned_[final_idx], d_current_loss, 
                               sizeof(float), cudaMemcpyDeviceToHost));
                               
        h_prev_loss = *h_loss_pinned_[final_idx];
        float final_loss = h_prev_loss / (batch_size * INPUT_H * INPUT_W * INPUT_C);
        epoch_loss += final_loss;

        cudaEventRecord(epoch_stop);
        cudaEventSynchronize(epoch_stop);

        float epoch_time;
        cudaEventElapsedTime(&epoch_time, epoch_start, epoch_stop);
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " - Loss: " << (epoch_loss / num_batches)
                  << " - Time: " << (epoch_time / 1000.0f) << "s" << std::endl;
        
    

        cudaEventDestroy(epoch_start);
        cudaEventDestroy(epoch_stop);
        
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    
    std::cout << "Training completed in " << (total_time / 1000.0f) << " seconds" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void AutoencoderGPUOptimized2::extract_features(const std::vector<float>& images,
                                                int num_images,
                                                std::vector<float>& features) {
    features.resize(num_images * LATENT_DIM);
    
    int batch_size = 64;  
    int num_batches = (num_images + batch_size - 1) / batch_size;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int batch = 0; batch < num_batches; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_images);
        int actual_batch_size = end_idx - start_idx;
        
        if (actual_batch_size != current_batch_size_) {
            allocate_device_memory(actual_batch_size);
        }
        
        const float* batch_data = &images[start_idx * INPUT_H * INPUT_W * INPUT_C];
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input_, batch_data,
                             actual_batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        launch_conv2d_forward_relu_fused(d_input_, d_conv1_out_, d_conv1_weights_, d_conv1_bias_,
                                   actual_batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
        
        launch_maxpool2d_optimized_forward(d_conv1_out_, d_pool1_out_, d_indices1_,
                                        actual_batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
        
        launch_conv2d_forward_relu_fused(d_pool1_out_, d_conv2_out_, d_conv2_weights_, d_conv2_bias_,
                                        actual_batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
        
        launch_maxpool2d_optimized_forward(d_conv2_out_, d_pool2_out_, d_indices2_,
                                    actual_batch_size, 16, 16, CONV2_FILTERS, 2, 2);
        
        CUDA_CHECK(cudaMemcpy(&features[start_idx * LATENT_DIM],
                             d_pool2_out_,
                             actual_batch_size * LATENT_DIM * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(end - start).count();
    
    std::cout << "Optimized feature extraction completed in " << time << " seconds" << std::endl;
}

void AutoencoderGPUOptimized2::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving: " << filepath << std::endl;
        return;
    }
    
    auto save_tensor = [&](float* d_ptr, int size) {
        std::vector<float> h_data(size);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(h_data.data()), size * sizeof(float));
    };
    
    save_tensor(d_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3);
    save_tensor(d_conv1_bias_, CONV1_FILTERS);
    save_tensor(d_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3);
    save_tensor(d_conv2_bias_, CONV2_FILTERS);
    save_tensor(d_conv3_weights_, LATENT_C * LATENT_C * 3 * 3);
    save_tensor(d_conv3_bias_, LATENT_C);
    save_tensor(d_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3);
    save_tensor(d_conv4_bias_, CONV1_FILTERS);
    save_tensor(d_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3);
    save_tensor(d_conv5_bias_, INPUT_C);
    
    file.close();
    std::cout << "Optimized weights saved to " << filepath << std::endl;
}

void AutoencoderGPUOptimized2::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading: " << filepath << std::endl;
        return;
    }
    
    auto load_tensor = [&](float* d_ptr, int expected_size) {
        int size;
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        if (size != expected_size) {
            std::cerr << "Size mismatch!" << std::endl;
            return;
        }
        std::vector<float> h_data(size);
        file.read(reinterpret_cast<char*>(h_data.data()), size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    };
    
    load_tensor(d_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3);
    load_tensor(d_conv1_bias_, CONV1_FILTERS);
    load_tensor(d_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3);
    load_tensor(d_conv2_bias_, CONV2_FILTERS);
    load_tensor(d_conv3_weights_, LATENT_C * LATENT_C * 3 * 3);
    load_tensor(d_conv3_bias_, LATENT_C);
    load_tensor(d_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3);
    load_tensor(d_conv4_bias_, CONV1_FILTERS);
    load_tensor(d_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3);
    load_tensor(d_conv5_bias_, INPUT_C);
    
    file.close();
    std::cout << "Optimized weights loaded from " << filepath << std::endl;
}

void AutoencoderGPUOptimized2::copy_weights_from_cpu(const std::string& cpu_weights_path) {
    load_weights(cpu_weights_path);
}
