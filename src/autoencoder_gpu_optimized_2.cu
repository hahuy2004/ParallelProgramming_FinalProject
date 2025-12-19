// Naive GPU Implementation
#include "../include/autoencoder_gpu_optimized_2.h"
#include "../cuda/gpu_kernels_optimized_2.h"
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
#include <random>

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

AutoencoderGPUOptimized2::AutoencoderGPUOptimized2() {
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);      // Fixed seed for reproducibility
    
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
    
    // Initialize conv1: 3 -> 256, kernel 3x3
    init_and_upload(&d_conv1_weight, CONV1_FILTERS * INPUT_C * 3 * 3, INPUT_C * 3 * 3);
    init_bias(&d_conv1_bias, CONV1_FILTERS);
    
    // Initialize conv2: 256 -> 128, kernel 3x3
    init_and_upload(&d_conv2_weight, CONV2_FILTERS * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv2_bias, CONV2_FILTERS);
    
    // Initialize conv3: 128 -> 128, kernel 3x3
    init_and_upload(&d_conv3_weight, LATENT_C * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv3_bias, LATENT_C);
    
    // Initialize conv4: 128 -> 256, kernel 3x3
    init_and_upload(&d_conv4_weight, CONV1_FILTERS * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv4_bias, CONV1_FILTERS);
    
    // Initialize conv5: 256 -> 3, kernel 3x3
    init_and_upload(&d_conv5_weight, INPUT_C * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv5_bias, INPUT_C);
    
    // Allocate loss buffer
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    
    // Allocate gradient buffers for weights
    CUDA_CHECK(cudaMalloc(&d_conv1_weight_grad, CONV1_FILTERS * INPUT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_bias_grad, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_weight_grad, CONV2_FILTERS * CONV1_FILTERS * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_bias_grad, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_weight_grad, LATENT_C * LATENT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_bias_grad, LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_weight_grad, CONV1_FILTERS * LATENT_C * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_bias_grad, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_weight_grad, INPUT_C * CONV1_FILTERS * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_bias_grad, INPUT_C * sizeof(float)));
    
    // Allocate device memory for activations
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu1_out, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_out, CONV1_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, CONV2_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu2_out, CONV2_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_out, LATENT_C * LATENT_H * LATENT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out, LATENT_C * LATENT_H * LATENT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu3_out, LATENT_C * LATENT_H * LATENT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_out, LATENT_C * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_out, CONV1_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu4_out, CONV1_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_out, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_out, INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    
    // Allocate device memory for gradient buffers
    CUDA_CHECK(cudaMalloc(&d_grad_conv5, INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up2, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu4, CONV1_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4, CONV1_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_up1, LATENT_C * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu3, LATENT_C * LATENT_H * LATENT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv3, LATENT_C * LATENT_H * LATENT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool2, LATENT_C * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu2, CONV2_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv2, CONV2_FILTERS * (INPUT_H/2) * (INPUT_W/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_pool1, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_relu1, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv1, CONV1_FILTERS * INPUT_H * INPUT_W * sizeof(float)));
    
    std::cout << "GPU weights initialized" << std::endl;
}

AutoencoderGPUOptimized2::~AutoencoderGPUOptimized2() {
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

float AutoencoderGPUOptimized2::train_step(const float* input_chw, float learning_rate) {
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

    // Backward pass
    backward();
    
    // Update weights
    update_weights(learning_rate);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return h_loss;
}

void AutoencoderGPUOptimized2::train(const std::vector<float>& train_images,
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

void AutoencoderGPUOptimized2::forward() {
    // Conv1: (3, 32, 32) -> (256, 32, 32)
    // kernel_size=3, stride=1, padding=1
        launch_conv2d_relu_forward(d_input, d_conv1_weight, d_conv1_bias, d_conv1_out,
                         3, 32, 32, 256, 32, 32, 3, 1, 1);
    
    // MaxPool1: (256, 32, 32) -> (256, 16, 16)
    // pool_size=2, stride=2
    launch_maxpool_unroll_forward(d_conv1_out, d_pool1_out, 256, 32, 32, 2, 2);
    
    // Conv2: (256, 16, 16) -> (128, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_relu_forward(d_pool1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                         256, 16, 16, 128, 16, 16, 3, 1, 1);
        
    // MaxPool2: (128, 16, 16) -> (128, 8, 8)
    // pool_size=2, stride=2
    launch_maxpool_unroll_forward(d_conv2_out, d_pool2_out, 128, 16, 16, 2, 2);
    
    // === DECODER ===
    
    // Conv3: (128, 8, 8) -> (128, 8, 8)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_relu_forward(d_pool2_out, d_conv3_weight, d_conv3_bias, d_conv3_out,
                         128, 8, 8, 128, 8, 8, 3, 1, 1);
    
    // Upsample1: (128, 8, 8) -> (128, 16, 16)
    // scale_factor=2
    launch_upsample_forward(d_conv3_out, d_up1_out, 128, 8, 8, 2);
    
    // Conv4: (128, 16, 16) -> (256, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_relu_forward(d_up1_out, d_conv4_weight, d_conv4_bias, d_conv4_out,
                         128, 16, 16, 256, 16, 16, 3, 1, 1);
        
    // Upsample2: (256, 16, 16) -> (256, 32, 32)
    // scale_factor=2
    launch_upsample_forward(d_conv4_out, d_up2_out, 256, 16, 16, 2);
    
    // Conv5: (256, 32, 32) -> (3, 32, 32) [No activation]
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_forward(d_up2_out, d_conv5_weight, d_conv5_bias, d_conv5_out,
                         256, 32, 32, 3, 32, 32, 3, 1, 1);
}

void AutoencoderGPUOptimized2::backward() {
    CUDA_CHECK(cudaMemset(d_grad_relu1, 0, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_relu2, 0, 128 * 16 * 16 * sizeof(float)));
    
    // Conv5 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv5, d_up2_out, d_conv5_weight_grad,
                             256, 32, 32, 3, 32, 32, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv5, d_conv5_bias_grad, 3, 32, 32);
    
    launch_conv2d_input_grad_unroll(d_grad_conv5, d_conv5_weight, d_grad_up2,
                            256, 32, 32, 3, 32, 32, 3, 1, 1);
    
    // Upsample2 backward - scale_factor=2
    launch_upsample_backward(d_grad_up2, d_grad_relu4, 256, 16, 16, 2);
    
    // ReLU4 backward
    launch_relu_backward(d_grad_relu4, d_conv4_out, d_grad_conv4, 256*16*16);
    
    // Conv4 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv4, d_up1_out, d_conv4_weight_grad,
                             128, 16, 16, 256, 16, 16, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv4, d_conv4_bias_grad, 256, 16, 16);
    
    launch_conv2d_input_grad_unroll(d_grad_conv4, d_conv4_weight, d_grad_up1,
                            128, 16, 16, 256, 16, 16, 3, 1, 1);
    
    // Upsample1 backward - scale_factor=2
    launch_upsample_backward(d_grad_up1, d_grad_relu3, 128, 8, 8, 2);
    
    // ReLU3 backward
    launch_relu_backward(d_grad_relu3, d_conv3_out, d_grad_conv3, 128*8*8);
    
    // Conv3 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv3, d_pool2_out, d_conv3_weight_grad,
                             128, 8, 8, 128, 8, 8, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv3, d_conv3_bias_grad, 128, 8, 8);
    
    launch_conv2d_input_grad_unroll(d_grad_conv3, d_conv3_weight, d_grad_pool2,
                            128, 8, 8, 128, 8, 8, 3, 1, 1);
    
    // MaxPool2 backward - pool_size=2, stride=2
    launch_maxpool_unroll_backward(d_grad_pool2, d_relu2_out, d_grad_relu2, 128, 16, 16, 2, 2);
    
    // ReLU2 backward
    launch_relu_backward(d_grad_relu2, d_conv2_out, d_grad_conv2, 128*16*16);
    
    // Conv2 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv2, d_pool1_out, d_conv2_weight_grad,
                             256, 16, 16, 128, 16, 16, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv2, d_conv2_bias_grad, 128, 16, 16);
    
    launch_conv2d_input_grad_unroll(d_grad_conv2, d_conv2_weight, d_grad_pool1,
                            256, 16, 16, 128, 16, 16, 3, 1, 1);
    
    // MaxPool1 backward - pool_size=2, stride=2
    launch_maxpool_unroll_backward(d_grad_pool1, d_relu1_out, d_grad_relu1, 256, 32, 32, 2, 2);
    
    // ReLU1 backward
    launch_relu_backward(d_grad_relu1, d_conv1_out, d_grad_conv1, 256*32*32);
    
    // Conv1 backward - kernel_size=3, stride=1, padding=1
    launch_conv2d_weight_grad(d_grad_conv1, d_input, d_conv1_weight_grad,
                             3, 32, 32, 256, 32, 32, 3, 1, 1);
    
    launch_conv2d_bias_grad(d_grad_conv1, d_conv1_bias_grad, 256, 32, 32);
}

void AutoencoderGPUOptimized2::update_weights(float learning_rate) {
    launch_sgd_update(d_conv1_weight, d_conv1_weight_grad, learning_rate, CONV1_FILTERS * INPUT_C * 3 * 3);
    launch_sgd_update(d_conv1_bias, d_conv1_bias_grad, learning_rate, CONV1_FILTERS);
    
    launch_sgd_update(d_conv2_weight, d_conv2_weight_grad, learning_rate, CONV2_FILTERS * CONV1_FILTERS * 3 * 3);
    launch_sgd_update(d_conv2_bias, d_conv2_bias_grad, learning_rate, CONV2_FILTERS);
    
    launch_sgd_update(d_conv3_weight, d_conv3_weight_grad, learning_rate, LATENT_C * LATENT_C * 3 * 3);
    launch_sgd_update(d_conv3_bias, d_conv3_bias_grad, learning_rate, LATENT_C);
    
    launch_sgd_update(d_conv4_weight, d_conv4_weight_grad, learning_rate, CONV1_FILTERS * LATENT_C * 3 * 3);
    launch_sgd_update(d_conv4_bias, d_conv4_bias_grad, learning_rate, CONV1_FILTERS);
    
    launch_sgd_update(d_conv5_weight, d_conv5_weight_grad, learning_rate, INPUT_C * CONV1_FILTERS * 3 * 3);
    launch_sgd_update(d_conv5_bias, d_conv5_bias_grad, learning_rate, INPUT_C);
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
    
    save_tensor(d_conv1_weight, CONV1_FILTERS * INPUT_C * 3 * 3);
    save_tensor(d_conv1_bias, CONV1_FILTERS);
    save_tensor(d_conv2_weight, CONV2_FILTERS * CONV1_FILTERS * 3 * 3);
    save_tensor(d_conv2_bias, CONV2_FILTERS);
    save_tensor(d_conv3_weight, LATENT_C * LATENT_C * 3 * 3);
    save_tensor(d_conv3_bias, LATENT_C);
    save_tensor(d_conv4_weight, CONV1_FILTERS * LATENT_C * 3 * 3);
    save_tensor(d_conv4_bias, CONV1_FILTERS);
    save_tensor(d_conv5_weight, INPUT_C * CONV1_FILTERS * 3 * 3);
    save_tensor(d_conv5_bias, INPUT_C);
    
    file.close();
    std::cout << "Weights saved to " << filepath << std::endl;
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
            std::cerr << "Size mismatch: expected " << expected_size << ", got " << size << std::endl;
            return;
        }
        std::vector<float> h_data(size);
        file.read(reinterpret_cast<char*>(h_data.data()), size * sizeof(float));
        if (!file.good()) {
            std::cerr << "Failed to read tensor data" << std::endl;
            return;
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    };
    
    load_tensor(d_conv1_weight, CONV1_FILTERS * INPUT_C * 3 * 3);
    load_tensor(d_conv1_bias, CONV1_FILTERS);
    load_tensor(d_conv2_weight, CONV2_FILTERS * CONV1_FILTERS * 3 * 3);
    load_tensor(d_conv2_bias, CONV2_FILTERS);
    load_tensor(d_conv3_weight, LATENT_C * LATENT_C * 3 * 3);
    load_tensor(d_conv3_bias, LATENT_C);
    load_tensor(d_conv4_weight, CONV1_FILTERS * LATENT_C * 3 * 3);
    load_tensor(d_conv4_bias, CONV1_FILTERS);
    load_tensor(d_conv5_weight, INPUT_C * CONV1_FILTERS * 3 * 3);
    load_tensor(d_conv5_bias, INPUT_C);
    
    file.close();
    std::cout << "Weights loaded from " << filepath << std::endl;
}
void AutoencoderGPUOptimized2::extract_features(const float* input_chw, float* output_features) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, 3*32*32*sizeof(float), cudaMemcpyHostToDevice));
    
    // Run encoder only (stop at bottleneck)
    
    // Conv1: (3, 32, 32) -> (256, 32, 32)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_relu_forward(d_input, d_conv1_weight, d_conv1_bias, d_conv1_out,
                         3, 32, 32, 256, 32, 32, 3, 1, 1);
        
    // MaxPool1: (256, 32, 32) -> (256, 16, 16)
    // pool_size=2, stride=2
    launch_maxpool_unroll_forward(d_conv1_out, d_pool1_out, 256, 32, 32, 2, 2);
    
    // Conv2: (256, 16, 16) -> (128, 16, 16)
    // kernel_size=3, stride=1, padding=1
    launch_conv2d_relu_forward(d_pool1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                         256, 16, 16, 128, 16, 16, 3, 1, 1);
        
    // MaxPool2: (128, 16, 16) -> (128, 8, 8) - BOTTLENECK/FEATURES
    // pool_size=2, stride=2
    launch_maxpool_forward(d_conv2_out, d_pool2_out, 128, 16, 16, 2, 2);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy features from device to host
    // d_pool2_out contains 128*8*8 = 8192 features
    CUDA_CHECK(cudaMemcpy(output_features, d_pool2_out, 128*8*8*sizeof(float), cudaMemcpyDeviceToHost));
}

void AutoencoderGPUOptimized2::reconstruct(const float* input_chw, float* output_reconstructed) {
    const int image_size = INPUT_C * INPUT_H * INPUT_W; // 3072 pixels
    // Copy input qua device 
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, image_size * sizeof(float), cudaMemcpyHostToDevice));
    // Chạy một lần forward 
    forward();
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy kết quả tái tạo từ device về host
    CUDA_CHECK(cudaMemcpy(output_reconstructed, d_conv5_out, image_size * sizeof(float), cudaMemcpyDeviceToHost));
}