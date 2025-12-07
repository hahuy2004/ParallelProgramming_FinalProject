#include "../include/autoencoder_gpu.h"
#include "../include/gpu_kernels.h"
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

AutoencoderGPU::AutoencoderGPU() : current_batch_size_(0) {
    // Initialize all pointers to nullptr
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
    d_conv2_out_ = nullptr;
    d_pool2_out_ = nullptr;
    d_conv3_out_ = nullptr;
    d_up1_out_ = nullptr;
    d_conv4_out_ = nullptr;
    d_up2_out_ = nullptr;
    d_conv5_out_ = nullptr;
    
    d_loss_ = nullptr;
    
    initialize_weights();
}

AutoencoderGPU::~AutoencoderGPU() {
    free_device_memory();
}

void AutoencoderGPU::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
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
    init_and_upload(&d_conv1_weights_, CONV1_FILTERS * INPUT_C * 3 * 3, INPUT_C * 3 * 3);
    init_bias(&d_conv1_bias_, CONV1_FILTERS);
    
    // Initialize conv2: 256 -> 128, kernel 3x3
    init_and_upload(&d_conv2_weights_, CONV2_FILTERS * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv2_bias_, CONV2_FILTERS);
    
    // Initialize conv3: 128 -> 128, kernel 3x3
    init_and_upload(&d_conv3_weights_, LATENT_C * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv3_bias_, LATENT_C);
    
    // Initialize conv4: 128 -> 256, kernel 3x3
    init_and_upload(&d_conv4_weights_, CONV1_FILTERS * LATENT_C * 3 * 3, LATENT_C * 3 * 3);
    init_bias(&d_conv4_bias_, CONV1_FILTERS);
    
    // Initialize conv5: 256 -> 3, kernel 3x3
    init_and_upload(&d_conv5_weights_, INPUT_C * CONV1_FILTERS * 3 * 3, CONV1_FILTERS * 3 * 3);
    init_bias(&d_conv5_bias_, INPUT_C);
    
    // Allocate loss buffer
    CUDA_CHECK(cudaMalloc(&d_loss_, sizeof(float)));
    
    std::cout << "GPU weights initialized" << std::endl;
}

void AutoencoderGPU::allocate_device_memory(int batch_size) {
    if (batch_size == current_batch_size_) return;
    
    // Free old memory if exists
    if (current_batch_size_ > 0) {
        cudaFree(d_input_);
        cudaFree(d_conv1_out_);
        cudaFree(d_pool1_out_);
        cudaFree(d_conv2_out_);
        cudaFree(d_pool2_out_);
        cudaFree(d_conv3_out_);
        cudaFree(d_up1_out_);
        cudaFree(d_conv4_out_);
        cudaFree(d_up2_out_);
        cudaFree(d_conv5_out_);
    }
    
    current_batch_size_ = batch_size;
    
    // Allocate activation buffers
    CUDA_CHECK(cudaMalloc(&d_input_, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out_, batch_size * 16 * 16 * CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out_, batch_size * LATENT_H * LATENT_W * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up1_out_, batch_size * 16 * 16 * LATENT_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_out_, batch_size * 16 * 16 * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up2_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_out_, batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
}

void AutoencoderGPU::free_device_memory() {
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
    if (d_conv2_out_) cudaFree(d_conv2_out_);
    if (d_pool2_out_) cudaFree(d_pool2_out_);
    if (d_conv3_out_) cudaFree(d_conv3_out_);
    if (d_up1_out_) cudaFree(d_up1_out_);
    if (d_conv4_out_) cudaFree(d_conv4_out_);
    if (d_up2_out_) cudaFree(d_up2_out_);
    if (d_conv5_out_) cudaFree(d_conv5_out_);
    
    if (d_loss_) cudaFree(d_loss_);
}

void AutoencoderGPU::forward_gpu(int batch_size) {
    // Encoder
    // Conv1 + ReLU: (32, 32, 3) -> (32, 32, 256)
    launch_conv2d_forward(d_input_, d_conv1_out_, d_conv1_weights_, d_conv1_bias_,
                         batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
    launch_relu_forward(d_conv1_out_, batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    
    // MaxPool: (32, 32, 256) -> (16, 16, 256)
    launch_maxpool2d_forward(d_conv1_out_, d_pool1_out_,
                            batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
    
    // Conv2 + ReLU: (16, 16, 256) -> (16, 16, 128)
    launch_conv2d_forward(d_pool1_out_, d_conv2_out_, d_conv2_weights_, d_conv2_bias_,
                         batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
    launch_relu_forward(d_conv2_out_, batch_size * 16 * 16 * CONV2_FILTERS);
    
    // MaxPool: (16, 16, 128) -> (8, 8, 128) - Latent
    launch_maxpool2d_forward(d_conv2_out_, d_pool2_out_,
                            batch_size, 16, 16, CONV2_FILTERS, 2, 2);
    
    // Decoder
    // Conv3 + ReLU: (8, 8, 128) -> (8, 8, 128)
    launch_conv2d_forward(d_pool2_out_, d_conv3_out_, d_conv3_weights_, d_conv3_bias_,
                         batch_size, LATENT_H, LATENT_W, LATENT_C, LATENT_C, 3, 1, 1);
    launch_relu_forward(d_conv3_out_, batch_size * LATENT_H * LATENT_W * LATENT_C);
    
    // Upsample: (8, 8, 128) -> (16, 16, 128)
    launch_upsample2d_forward(d_conv3_out_, d_up1_out_,
                             batch_size, LATENT_H, LATENT_W, LATENT_C, 2);
    
    // Conv4 + ReLU: (16, 16, 128) -> (16, 16, 256)
    launch_conv2d_forward(d_up1_out_, d_conv4_out_, d_conv4_weights_, d_conv4_bias_,
                         batch_size, 16, 16, LATENT_C, CONV1_FILTERS, 3, 1, 1);
    launch_relu_forward(d_conv4_out_, batch_size * 16 * 16 * CONV1_FILTERS);
    
    // Upsample: (16, 16, 256) -> (32, 32, 256)
    launch_upsample2d_forward(d_conv4_out_, d_up2_out_,
                             batch_size, 16, 16, CONV1_FILTERS, 2);
    
    // Conv5 (no activation): (32, 32, 256) -> (32, 32, 3)
    launch_conv2d_forward(d_up2_out_, d_conv5_out_, d_conv5_weights_, d_conv5_bias_,
                         batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_C, 3, 1, 1);
}

float AutoencoderGPU::compute_loss_gpu(int batch_size) {
    int size = batch_size * INPUT_H * INPUT_W * INPUT_C;
    launch_mse_loss(d_input_, d_conv5_out_, d_loss_, size);
    
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss / size;
}

void AutoencoderGPU::train(const std::vector<float>& train_images,
                            int num_images,
                            int batch_size,
                            int epochs,
                            float learning_rate) {
    std::cout << "Training GPU Autoencoder (Naive)..." << std::endl;
    std::cout << "Images: " << num_images << ", Batch size: " << batch_size 
              << ", Epochs: " << epochs << ", LR: " << learning_rate << std::endl;
    
    allocate_device_memory(batch_size);
    
    int num_batches = (num_images + batch_size - 1) / batch_size;
    
    // Create CUDA events for timing
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
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_images);
            int actual_batch_size = end_idx - start_idx;
            
            if (actual_batch_size != current_batch_size_) {
                allocate_device_memory(actual_batch_size);
            }
            
            const float* batch_data = &train_images[start_idx * INPUT_H * INPUT_W * INPUT_C];
            
            // Copy input to device
            CUDA_CHECK(cudaMemcpy(d_input_, batch_data,
                                 actual_batch_size * INPUT_H * INPUT_W * INPUT_C * sizeof(float),
                                 cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_gpu(actual_batch_size);
            
            // Compute loss
            float loss = compute_loss_gpu(actual_batch_size);
            epoch_loss += loss;
            
            // Note: Backward pass and weight updates omitted for naive version
            // In a complete implementation, you would add gradient computation here
        }
        
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

void AutoencoderGPU::extract_features(const std::vector<float>& images,
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
        
        // Run encoder only
        launch_conv2d_forward(d_input_, d_conv1_out_, d_conv1_weights_, d_conv1_bias_,
                             actual_batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
        launch_relu_forward(d_conv1_out_, actual_batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
        
        launch_maxpool2d_forward(d_conv1_out_, d_pool1_out_,
                                actual_batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
        
        launch_conv2d_forward(d_pool1_out_, d_conv2_out_, d_conv2_weights_, d_conv2_bias_,
                             actual_batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
        launch_relu_forward(d_conv2_out_, actual_batch_size * 16 * 16 * CONV2_FILTERS);
        
        launch_maxpool2d_forward(d_conv2_out_, d_pool2_out_,
                                actual_batch_size, 16, 16, CONV2_FILTERS, 2, 2);
        
        // Copy latent features back to host
        CUDA_CHECK(cudaMemcpy(&features[start_idx * LATENT_DIM],
                             d_pool2_out_,
                             actual_batch_size * LATENT_DIM * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(end - start).count();
    
    std::cout << "Feature extraction completed in " << time << " seconds" << std::endl;
}

void AutoencoderGPU::save_weights(const std::string& filepath) {
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
    std::cout << "Weights saved to " << filepath << std::endl;
}

void AutoencoderGPU::load_weights(const std::string& filepath) {
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
    std::cout << "Weights loaded from " << filepath << std::endl;
}

void AutoencoderGPU::copy_weights_from_cpu(const std::string& cpu_weights_path) {
    load_weights(cpu_weights_path);
}
