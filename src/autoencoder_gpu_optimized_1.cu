// Naive GPU Implementation
#include "../include/autoencoder_gpu_optimized_1.h"
#include "../cuda/gpu_kernels_optimized_1.h"
#include "../cuda/gpu_kernels_naive.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// AUTOENCODER CLASS IMPLEMENTATION
// ============================================================================

AutoencoderGPUOptimized1::AutoencoderGPUOptimized1() : last_loss(0.0f) {
    // Initialize weights on host
    h_conv1_weight.resize(256 * 3 * 3 * 3);
    h_conv1_bias.resize(256);
    h_conv2_weight.resize(128 * 256 * 3 * 3);
    h_conv2_bias.resize(128);
    h_conv3_weight.resize(128 * 128 * 3 * 3);
    h_conv3_bias.resize(128);
    h_conv4_weight.resize(256 * 128 * 3 * 3);
    h_conv4_bias.resize(256);
    h_conv5_weight.resize(3 * 256 * 3 * 3);
    h_conv5_bias.resize(3);
    
    // Simple random initialization (same as CPU)
    srand(42);  // Fixed seed for reproducibility
    for (auto& w : h_conv1_weight) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : h_conv2_weight) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : h_conv3_weight) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : h_conv4_weight) w = ((rand() % 100) / 500.0f - 0.1f);
    for (auto& w : h_conv5_weight) w = ((rand() % 100) / 500.0f - 0.1f);
    
    std::fill(h_conv1_bias.begin(), h_conv1_bias.end(), 0.0f);
    std::fill(h_conv2_bias.begin(), h_conv2_bias.end(), 0.0f);
    std::fill(h_conv3_bias.begin(), h_conv3_bias.end(), 0.0f);
    std::fill(h_conv4_bias.begin(), h_conv4_bias.end(), 0.0f);
    std::fill(h_conv5_bias.begin(), h_conv5_bias.end(), 0.0f);
    
    // Allocate device memory for weights
    CUDA_CHECK(cudaMalloc(&d_conv1_weight, h_conv1_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias, h_conv1_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_weight, h_conv2_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias, h_conv2_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_weight, h_conv3_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_bias, h_conv3_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_weight, h_conv4_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_bias, h_conv4_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_weight, h_conv5_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_bias, h_conv5_bias.size() * sizeof(float)));
    
    // Copy weights to device
    CUDA_CHECK(cudaMemcpy(d_conv1_weight, h_conv1_weight.data(), h_conv1_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, h_conv1_bias.data(), h_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_weight, h_conv2_weight.data(), h_conv2_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, h_conv2_bias.data(), h_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv3_weight, h_conv3_weight.data(), h_conv3_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv3_bias, h_conv3_bias.data(), h_conv3_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv4_weight, h_conv4_weight.data(), h_conv4_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv4_bias, h_conv4_bias.data(), h_conv4_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv5_weight, h_conv5_weight.data(), h_conv5_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv5_bias, h_conv5_bias.data(), h_conv5_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&d_conv1_weight_grad, h_conv1_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias_grad, h_conv1_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_weight_grad, h_conv2_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias_grad, h_conv2_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_weight_grad, h_conv3_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_bias_grad, h_conv3_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_weight_grad, h_conv4_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_bias_grad, h_conv4_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_weight_grad, h_conv5_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_bias_grad, h_conv5_bias.size() * sizeof(float)));
    
    // Allocate device memory for activations
    CUDA_CHECK(cudaMalloc(&d_input, 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu1_out, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_out, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu2_out, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_out, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu3_out, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_out, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_out, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu4_out, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_out, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_out, 3 * 32 * 32 * sizeof(float)));
    
    // Allocate device memory for gradient buffers
    CUDA_CHECK(cudaMalloc(&d_grad_conv5, 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up2, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu4, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up1, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu3, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv3, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool2, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu2, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv2, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool1, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu1, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv1, 256 * 32 * 32 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    std::cout << "GPU Autoencoder initialized (Memory Optimization)" << std::endl;
}

AutoencoderGPUOptimized1::~AutoencoderGPUOptimized1() {
    // Free all device memory
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_conv3_weight);
    cudaFree(d_conv3_bias);
    cudaFree(d_conv4_weight);
    cudaFree(d_conv4_bias);
    cudaFree(d_conv5_weight);
    cudaFree(d_conv5_bias);
    
    cudaFree(d_conv1_weight_grad);
    cudaFree(d_conv1_bias_grad);
    cudaFree(d_conv2_weight_grad);
    cudaFree(d_conv2_bias_grad);
    cudaFree(d_conv3_weight_grad);
    cudaFree(d_conv3_bias_grad);
    cudaFree(d_conv4_weight_grad);
    cudaFree(d_conv4_bias_grad);
    cudaFree(d_conv5_weight_grad);
    cudaFree(d_conv5_bias_grad);
    
    cudaFree(d_input);
    cudaFree(d_conv1_out);
    cudaFree(d_relu1_out);
    cudaFree(d_pool1_out);
    cudaFree(d_conv2_out);
    cudaFree(d_relu2_out);
    cudaFree(d_pool2_out);
    cudaFree(d_conv3_out);
    cudaFree(d_relu3_out);
    cudaFree(d_up1_out);
    cudaFree(d_conv4_out);
    cudaFree(d_relu4_out);
    cudaFree(d_up2_out);
    cudaFree(d_conv5_out);
    
    cudaFree(d_grad_conv5);
    cudaFree(d_grad_up2);
    cudaFree(d_grad_relu4);
    cudaFree(d_grad_conv4);
    cudaFree(d_grad_up1);
    cudaFree(d_grad_relu3);
    cudaFree(d_grad_conv3);
    cudaFree(d_grad_pool2);
    cudaFree(d_grad_relu2);
    cudaFree(d_grad_conv2);
    cudaFree(d_grad_pool1);
    cudaFree(d_grad_relu1);
    cudaFree(d_grad_conv1);
    
    cudaFree(d_loss);
}

float AutoencoderGPUOptimized1::train_step(const float* input_chw, float learning_rate) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, 3*32*32*sizeof(float), cudaMemcpyHostToDevice));

    // Forward pass
    forward();
    
    // Compute loss and gradient
    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    int size = 3 * 32 * 32;
    launch_mse_loss(d_conv5_out, d_input, d_loss, d_grad_conv5, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    last_loss = h_loss;
    
    // Backward pass
    backward();
    
    // Update weights
    update_weights(learning_rate);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return h_loss;
}

void AutoencoderGPUOptimized1::train(const std::vector<float>& train_images,
                          int num_train_images,
                          int batch_size,
                          int epochs,
                          float learning_rate) {
    const int IMAGE_PIXELS = 3072;  // 32*32*3
    
    // Calculate total batches
    int total_batches = (num_train_images + batch_size - 1) / batch_size;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    float total_images_processed = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        
        float epoch_loss = 0.0f;
        int seen = 0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        for (int start = 0; start < num_train_images; start += batch_size) {
            int end = std::min(start + batch_size, num_train_images);
            float batch_loss = 0.0f;
            
            // Train on each image in batch
            for (int i = start; i < end; i++) {
                const float* img_data = &train_images[i * IMAGE_PIXELS];
                batch_loss += train_step(img_data, learning_rate);
            }
            
            batch_loss /= (end - start);
            epoch_loss += batch_loss;
            seen++;
            total_images_processed += (end - start);
            
            int batch_num = start / batch_size + 1;
            std::cout << "Batch [" << batch_num << "/" << total_batches << "] Avg Loss: " 
                      << batch_loss << std::endl;
        }
        
        epoch_loss /= seen;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch avg loss: " << epoch_loss << " | time: " << epoch_time << "s" << std::endl;
    }
    
    // Training completed summary
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(total_end - total_start).count();
    int total_minutes = static_cast<int>(total_time) / 60;
    int total_secs = static_cast<int>(total_time) % 60;
    float avg_epoch_time = total_time / epochs;
    float overall_throughput = total_images_processed / total_time;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "         TRAINING COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  Total Time         : " << total_minutes << "m " << total_secs << "s (" 
              << std::fixed << std::setprecision(2) << total_time << "s)" << std::endl;
    std::cout << "  Avg Time/Epoch     : " << avg_epoch_time << "s" << std::endl;
    std::cout << "  Overall Throughput : " << std::setprecision(1) << overall_throughput << " images/sec" << std::endl;
    std::cout << "  Total Images       : " << static_cast<int>(total_images_processed) << " (" 
              << num_train_images << " × " << epochs << " epochs)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void AutoencoderGPUOptimized1::forward() {
    // Conv1: (3, 32, 32) -> (256, 32, 32)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_input, d_conv1_weight, d_conv1_bias, d_conv1_out,
                         INPUT_C, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_H, INPUT_W, 3, 1, 1);
    
    // ReLU1
    launch_relu_forward(d_conv1_out, d_relu1_out, 256*32*32);
    
    // MaxPool1: (256, 32, 32) -> (256, 16, 16)
    // pool_size=2, stride=2
    launch_maxpool_forward(d_relu1_out, d_pool1_out, 256, 32, 32, 2, 2);
    
    // Conv2: (256, 16, 16) -> (128, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_pool1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                         CONV1_FILTERS, INPUT_H/2, INPUT_W/2, CONV2_FILTERS, INPUT_H/2, INPUT_W/2, 3, 1, 1);
    
    // ReLU2
    launch_relu_forward(d_conv2_out, d_relu2_out, 128*16*16);
    
    // MaxPool2: (128, 16, 16) -> (128, 8, 8)
    // pool_size=2, stride=2
    launch_maxpool_forward(d_relu2_out, d_pool2_out, 128, 16, 16, 2, 2);
    
    // === DECODER ===
    
    // Conv3: (128, 8, 8) -> (128, 8, 8)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_pool2_out, d_conv3_weight, d_conv3_bias, d_conv3_out,
                         LATENT_C, LATENT_H, LATENT_W, LATENT_C, LATENT_H, LATENT_W, 3, 1, 1);
    
    // ReLU3
    launch_relu_forward(d_conv3_out, d_relu3_out, 128*8*8);
    
    // Upsample1: (128, 8, 8) -> (128, 16, 16)
    // scale_factor=2
    launch_upsample_forward(d_relu3_out, d_up1_out, 128, 8, 8, 2);
    
    // Conv4: (128, 16, 16) -> (256, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_up1_out, d_conv4_weight, d_conv4_bias, d_conv4_out,
                         LATENT_C, INPUT_H/2, INPUT_W/2, CONV1_FILTERS, INPUT_H/2, INPUT_W/2, 3, 1, 1);
    
    // ReLU4
    launch_relu_forward(d_conv4_out, d_relu4_out, 256*16*16);
    
    // Upsample2: (256, 16, 16) -> (256, 32, 32)
    // scale_factor=2
    launch_upsample_forward(d_relu4_out, d_up2_out, 256, 16, 16, 2);
    
    // Conv5: (256, 32, 32) -> (3, 32, 32) [No activation]
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_up2_out, d_conv5_weight, d_conv5_bias, d_conv5_out,
                         CONV1_FILTERS, INPUT_H, INPUT_W, INPUT_C, INPUT_H, INPUT_W, 3, 1, 1);
}

void AutoencoderGPUOptimized1::backward() {
    CUDA_CHECK(cudaMemset(d_grad_relu1, 0, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_relu2, 0, 128 * 16 * 16 * sizeof(float)));
    
    // Conv5 backward - kernel_size=3, stride=1, padding=1    
    launch_conv2d_shared_backward(d_grad_conv5, d_up2_out, d_grad_conv5, d_grad_up2, d_conv5_weight, d_conv5_bias_grad, 
                                     CONV1_FILTERS, INPUT_H, INPUT_W, INPUT_C, INPUT_H, INPUT_W, 3, 1, 1);

    // Upsample2 backward - scale_factor=2
    launch_upsample_backward(d_grad_up2, d_grad_relu4, 256, 16, 16, 2);
    
    // ReLU4 backward
    launch_relu_backward(d_grad_relu4, d_conv4_out, d_grad_conv4, 256*16*16);
    
    // Conv4 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_backward(d_grad_conv4, d_up1_out, d_conv4_weight, d_grad_up1, 
        d_conv4_weight_grad, d_conv4_bias_grad, 
        LATENT_C, INPUT_H/2, INPUT_W/2, CONV1_FILTERS, INPUT_H/2, INPUT_W/2, 3, 1, 1);
    
    // Upsample1 backward - scale_factor=2
    launch_upsample_backward(d_grad_up1, d_grad_relu3, 128, 8, 8, 2);
    
    // ReLU3 backward
    launch_relu_backward(d_grad_relu3, d_conv3_out, d_grad_conv3, 128*8*8);
    
    // Conv3 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_backward(d_grad_conv3, d_pool2_out, d_conv3_weight, d_grad_pool2,
                           d_conv3_weight_grad, d_conv3_bias_grad,
                           LATENT_C, LATENT_H, LATENT_W, LATENT_C, LATENT_H, LATENT_W, 3, 1, 1);
    
    // MaxPool2 backward - pool_size=2, stride=2
    launch_maxpool_backward(d_grad_pool2, d_relu2_out, d_grad_relu2, 128, 16, 16, 2, 2);
    
    // ReLU2 backward
    launch_relu_backward(d_grad_relu2, d_conv2_out, d_grad_conv2, 128*16*16);
    
    // Conv2 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv2, d_pool1_out, d_conv2_weight_grad,
                             256, 16, 16, 128, 16, 16, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv2, d_conv2_bias_grad, 128, 16, 16);
    
    launch_conv2d_input_grad(d_grad_conv2, d_conv2_weight, d_grad_pool1,
                            256, 16, 16, 128, 16, 16, 3, 1, 1);

    launch_conv2d_shared_backward(d_grad_conv2, d_pool1_out, d_conv2_weight, d_grad_pool1,
                            d_conv2_weight_grad, d_conv2_bias_grad,
                            CONV1_FILTERS, INPUT_H/2, INPUT_W/2, CONV2_FILTERS, INPUT_H/2, INPUT_W/2, 3, 1, 1);
    
    // MaxPool1 backward - pool_size=2, stride=2
    launch_maxpool_backward(d_grad_pool1, d_relu1_out, d_grad_relu1, 256, 32, 32, 2, 2);
    
    // ReLU1 backward
    launch_relu_backward(d_grad_relu1, d_conv1_out, d_grad_conv1, 256*32*32);
    
    // Conv1 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv1, d_input, d_conv1_weight_grad,
                             INPUT_C, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_H, INPUT_W, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv1, d_conv1_bias_grad, CONV1_FILTERS, INPUT_H, INPUT_W);
}

void AutoencoderGPUOptimized1::update_weights(float learning_rate) {
    launch_sgd_update(d_conv1_weight, d_conv1_weight_grad, learning_rate, h_conv1_weight.size());
    launch_sgd_update(d_conv1_bias, d_conv1_bias_grad, learning_rate, h_conv1_bias.size());
    
    launch_sgd_update(d_conv2_weight, d_conv2_weight_grad, learning_rate, h_conv2_weight.size());
    launch_sgd_update(d_conv2_bias, d_conv2_bias_grad, learning_rate, h_conv2_bias.size());
    
    launch_sgd_update(d_conv3_weight, d_conv3_weight_grad, learning_rate, h_conv3_weight.size());
    launch_sgd_update(d_conv3_bias, d_conv3_bias_grad, learning_rate, h_conv3_bias.size());
    
    launch_sgd_update(d_conv4_weight, d_conv4_weight_grad, learning_rate, h_conv4_weight.size());
    launch_sgd_update(d_conv4_bias, d_conv4_bias_grad, learning_rate, h_conv4_bias.size());
    
    launch_sgd_update(d_conv5_weight, d_conv5_weight_grad, learning_rate, h_conv5_weight.size());
    launch_sgd_update(d_conv5_bias, d_conv5_bias_grad, learning_rate, h_conv5_bias.size());
}

bool AutoencoderGPUOptimized1::save_weights(const std::string& filepath) const {
    // Copy weights from device to host
    CUDA_CHECK(cudaMemcpy(h_conv1_weight.data(), d_conv1_weight, h_conv1_weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv1_bias.data(), d_conv1_bias, h_conv1_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv2_weight.data(), d_conv2_weight, h_conv2_weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv2_bias.data(), d_conv2_bias, h_conv2_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv3_weight.data(), d_conv3_weight, h_conv3_weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv3_bias.data(), d_conv3_bias, h_conv3_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv4_weight.data(), d_conv4_weight, h_conv4_weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv4_bias.data(), d_conv4_bias, h_conv4_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv5_weight.data(), d_conv5_weight, h_conv5_weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv5_bias.data(), d_conv5_bias, h_conv5_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write((const char*)h_conv1_weight.data(), h_conv1_weight.size() * sizeof(float));
    file.write((const char*)h_conv1_bias.data(), h_conv1_bias.size() * sizeof(float));
    file.write((const char*)h_conv2_weight.data(), h_conv2_weight.size() * sizeof(float));
    file.write((const char*)h_conv2_bias.data(), h_conv2_bias.size() * sizeof(float));
    file.write((const char*)h_conv3_weight.data(), h_conv3_weight.size() * sizeof(float));
    file.write((const char*)h_conv3_bias.data(), h_conv3_bias.size() * sizeof(float));
    file.write((const char*)h_conv4_weight.data(), h_conv4_weight.size() * sizeof(float));
    file.write((const char*)h_conv4_bias.data(), h_conv4_bias.size() * sizeof(float));
    file.write((const char*)h_conv5_weight.data(), h_conv5_weight.size() * sizeof(float));
    file.write((const char*)h_conv5_bias.data(), h_conv5_bias.size() * sizeof(float));
    
    file.close();
    return true;
}

bool AutoencoderGPUOptimized1::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weights file: " << filepath << std::endl;
        return false;
    }

    auto read_or_fail = [&](float* ptr, size_t n) -> bool {
        file.read(reinterpret_cast<char*>(ptr), n * sizeof(float));
        return file.good();
    };

    // Ensure host buffers have correct sizes
    if (h_conv1_weight.size() != 256 * 3 * 3 * 3)  h_conv1_weight.resize(256 * 3 * 3 * 3);
    if (h_conv1_bias.size() != 256)              h_conv1_bias.resize(256);
    if (h_conv2_weight.size() != 128 * 256 * 3 * 3) h_conv2_weight.resize(128 * 256 * 3 * 3);
    if (h_conv2_bias.size() != 128)               h_conv2_bias.resize(128);
    if (h_conv3_weight.size() != 128 * 128 * 3 * 3) h_conv3_weight.resize(128 * 128 * 3 * 3);
    if (h_conv3_bias.size() != 128)               h_conv3_bias.resize(128);
    if (h_conv4_weight.size() != 256 * 128 * 3 * 3) h_conv4_weight.resize(256 * 128 * 3 * 3);
    if (h_conv4_bias.size() != 256)               h_conv4_bias.resize(256);
    if (h_conv5_weight.size() != 3 * 256 * 3 * 3)   h_conv5_weight.resize(3 * 256 * 3 * 3);
    if (h_conv5_bias.size() != 3)                 h_conv5_bias.resize(3);

    // Read in EXACT order matching save_weights()
    if (!read_or_fail(h_conv1_weight.data(), h_conv1_weight.size())) return false;
    if (!read_or_fail(h_conv1_bias.data(), h_conv1_bias.size())) return false;
    if (!read_or_fail(h_conv2_weight.data(), h_conv2_weight.size())) return false;
    if (!read_or_fail(h_conv2_bias.data(), h_conv2_bias.size())) return false;
    if (!read_or_fail(h_conv3_weight.data(), h_conv3_weight.size())) return false;
    if (!read_or_fail(h_conv3_bias.data(), h_conv3_bias.size())) return false;
    if (!read_or_fail(h_conv4_weight.data(), h_conv4_weight.size())) return false;
    if (!read_or_fail(h_conv4_bias.data(), h_conv4_bias.size())) return false;
    if (!read_or_fail(h_conv5_weight.data(), h_conv5_weight.size())) return false;
    if (!read_or_fail(h_conv5_bias.data(), h_conv5_bias.size())) return false;

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_conv1_weight, h_conv1_weight.data(), h_conv1_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, h_conv1_bias.data(), h_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_weight, h_conv2_weight.data(), h_conv2_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, h_conv2_bias.data(), h_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv3_weight, h_conv3_weight.data(), h_conv3_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv3_bias, h_conv3_bias.data(), h_conv3_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv4_weight, h_conv4_weight.data(), h_conv4_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv4_bias, h_conv4_bias.data(), h_conv4_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv5_weight, h_conv5_weight.data(), h_conv5_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv5_bias, h_conv5_bias.data(), h_conv5_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Loaded weights from: " << filepath << std::endl;
    return true;
}

void AutoencoderGPUOptimized1::extract_features(const float* input_chw, float* output_features) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, 3*32*32*sizeof(float), cudaMemcpyHostToDevice));
    
    // Run encoder only (stop at bottleneck)
    
    // Conv1: (3, 32, 32) -> (256, 32, 32)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_input, d_conv1_weight, d_conv1_bias, d_conv1_out,
                         INPUT_C, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_H, INPUT_W, 3, 1, 1);

    // ReLU1
    launch_relu_forward(d_conv1_out, d_relu1_out, 256*32*32);
    
    // MaxPool1: (256, 32, 32) -> (256, 16, 16)
    // pool_size=2, stride=2
    launch_maxpool_forward(d_relu1_out, d_pool1_out, 256, 32, 32, 2, 2);
    
    // Conv2: (256, 16, 16) -> (128, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_shared_forward(d_pool1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                        CONV1_FILTERS, INPUT_H/2, INPUT_W/2, CONV2_FILTERS, INPUT_H/2, INPUT_W/2, 3, 1, 1);
    
    // ReLU2
    launch_relu_forward(d_conv2_out, d_relu2_out, 128*16*16);
    
    // MaxPool2: (128, 16, 16) -> (128, 8, 8) - BOTTLENECK/FEATURES
    // pool_size=2, stride=2
    launch_maxpool_forward(d_relu2_out, d_pool2_out, 128, 16, 16, 2, 2);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy features from device to host
    // d_pool2_out contains 128*8*8 = 8192 features
    CUDA_CHECK(cudaMemcpy(output_features, d_pool2_out, 128*8*8*sizeof(float), cudaMemcpyDeviceToHost));
}
