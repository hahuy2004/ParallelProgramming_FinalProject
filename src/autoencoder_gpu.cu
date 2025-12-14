#include "autoencoder.h"
#include "gpu_kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// ConvLayer Implementation
// ============================================================================

ConvLayer::ConvLayer(int in_channels, int out_channels, int k_size) 
    : in_c(in_channels), out_c(out_channels), kernel_size(k_size) {
    
    weight_size = out_c * in_c * k_size * k_size;
    bias_size = out_c;
    
    CUDA_CHECK(cudaMalloc(&d_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, bias_size * sizeof(float)));
    
    // Xavier initialization
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_bias(bias_size, 0.0f);
    float std = sqrtf(2.0f / (in_c * k_size * k_size));
    
    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), 
                         weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 
                         bias_size * sizeof(float), cudaMemcpyHostToDevice));
}

ConvLayer::~ConvLayer() {
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
}

// ============================================================================
// Autoencoder Implementation
// ============================================================================

Autoencoder::Autoencoder(int batch, float lr) 
    : batch_size(batch), learning_rate(lr) {
    
    // Initialize layers
    conv1 = new ConvLayer(3, 256, 3);
    conv2 = new ConvLayer(256, 128, 3);
    conv3 = new ConvLayer(128, 128, 3);
    conv4 = new ConvLayer(128, 256, 3);
    conv5 = new ConvLayer(256, 3, 3);
    
    // Allocate memory for encoder activations
    CUDA_CHECK(cudaMalloc(&d_enc1, batch * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc1_relu, batch * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc1_pool, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_indices, batch * 16 * 16 * 256 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_enc2, batch * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc2_relu, batch * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_latent, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool2_indices, batch * 8 * 8 * 128 * sizeof(float)));
    
    // Allocate memory for decoder activations
    CUDA_CHECK(cudaMalloc(&d_dec1, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec1_relu, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec1_up, batch * 16 * 16 * 128 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_dec2, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec2_relu, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec2_up, batch * 32 * 32 * 256 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_output, batch * 32 * 32 * 3 * sizeof(float)));
    
    // Allocate gradients
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch * 32 * 32 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec2_up, batch * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec2_relu, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec2, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec1_up, batch * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec1_relu, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_dec1, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_latent, batch * 8 * 8 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc2_relu, batch * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc2, batch * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc1_pool, batch * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc1_relu, batch * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_enc1, batch * 32 * 32 * 256 * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
}

Autoencoder::~Autoencoder() {
    delete conv1;
    delete conv2;
    delete conv3;
    delete conv4;
    delete conv5;
    
    cudaFree(d_enc1);
    cudaFree(d_enc1_relu);
    cudaFree(d_enc1_pool);
    cudaFree(d_pool1_indices);
    cudaFree(d_enc2);
    cudaFree(d_enc2_relu);
    cudaFree(d_latent);
    cudaFree(d_pool2_indices);
    cudaFree(d_dec1);
    cudaFree(d_dec1_relu);
    cudaFree(d_dec1_up);
    cudaFree(d_dec2);
    cudaFree(d_dec2_relu);
    cudaFree(d_dec2_up);
    cudaFree(d_output);
    
    cudaFree(d_grad_output);
    cudaFree(d_grad_dec2_up);
    cudaFree(d_grad_dec2_relu);
    cudaFree(d_grad_dec2);
    cudaFree(d_grad_dec1_up);
    cudaFree(d_grad_dec1_relu);
    cudaFree(d_grad_dec1);
    cudaFree(d_grad_latent);
    cudaFree(d_grad_enc2_relu);
    cudaFree(d_grad_enc2);
    cudaFree(d_grad_enc1_pool);
    cudaFree(d_grad_enc1_relu);
    cudaFree(d_grad_enc1);
    cudaFree(d_loss);
}

float Autoencoder::forward(const float* d_input, const float* d_target) {
    // ========== ENCODER ==========
    
    // Conv1: (32,32,3) -> (32,32,256)
    launch_conv2d_forward(d_input, d_enc1, conv1->d_weights, conv1->d_bias,
                         batch_size, 32, 32, 3, 256, 3, 1, 1);
    
    // ReLU activation
    launch_relu_forward(d_enc1, batch_size * 32 * 32 * 256);
    CUDA_CHECK(cudaMemcpy(d_enc1_relu, d_enc1, 
                         batch_size * 32 * 32 * 256 * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // MaxPool1: (32,32,256) -> (16,16,256)
    launch_maxpool2d_forward(d_enc1_relu, d_enc1_pool, d_pool1_indices,
                            batch_size, 32, 32, 256, 2, 2);
    
    // Conv2: (16,16,256) -> (16,16,128)
    launch_conv2d_forward(d_enc1_pool, d_enc2, conv2->d_weights, conv2->d_bias,
                         batch_size, 16, 16, 256, 128, 3, 1, 1);
    
    // ReLU activation
    launch_relu_forward(d_enc2, batch_size * 16 * 16 * 128);
    CUDA_CHECK(cudaMemcpy(d_enc2_relu, d_enc2, 
                         batch_size * 16 * 16 * 128 * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // MaxPool2: (16,16,128) -> (8,8,128) [LATENT REPRESENTATION]
    launch_maxpool2d_forward(d_enc2_relu, d_latent, d_pool2_indices,
                            batch_size, 16, 16, 128, 2, 2);
    
    // ========== DECODER ==========
    
    // Conv3: (8,8,128) -> (8,8,128)
    launch_conv2d_forward(d_latent, d_dec1, conv3->d_weights, conv3->d_bias,
                         batch_size, 8, 8, 128, 128, 3, 1, 1);
    
    // ReLU activation
    launch_relu_forward(d_dec1, batch_size * 8 * 8 * 128);
    CUDA_CHECK(cudaMemcpy(d_dec1_relu, d_dec1, 
                         batch_size * 8 * 8 * 128 * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Upsample1: (8,8,128) -> (16,16,128)
    launch_upsample2d_forward(d_dec1_relu, d_dec1_up,
                             batch_size, 8, 8, 128, 2);
    
    // Conv4: (16,16,128) -> (16,16,256)
    launch_conv2d_forward(d_dec1_up, d_dec2, conv4->d_weights, conv4->d_bias,
                         batch_size, 16, 16, 128, 256, 3, 1, 1);
    
    // ReLU activation
    launch_relu_forward(d_dec2, batch_size * 16 * 16 * 256);
    CUDA_CHECK(cudaMemcpy(d_dec2_relu, d_dec2, 
                         batch_size * 16 * 16 * 256 * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Upsample2: (16,16,256) -> (32,32,256)
    launch_upsample2d_forward(d_dec2_relu, d_dec2_up,
                             batch_size, 16, 16, 256, 2);
    
    // Conv5: (32,32,256) -> (32,32,3) [OUTPUT - No activation]
    launch_conv2d_forward(d_dec2_up, d_output, conv5->d_weights, conv5->d_bias,
                         batch_size, 32, 32, 256, 3, 3, 1, 1);
    
    // Compute MSE loss
    int size = batch_size * 32 * 32 * 3;
    launch_mse_loss(d_output, d_target, d_loss, size);
    
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return h_loss;
}

void Autoencoder::backward(const float* d_input, const float* d_target) {
    // Compute gradient of loss with respect to output
    int size = batch_size * 32 * 32 * 3;
    launch_mse_loss_backward(d_output, d_target, d_grad_output, size);
    
    // ========== DECODER BACKWARD ==========
    
    // Backward through Conv5
    launch_conv2d_backward(d_grad_output, d_dec2_up, conv5->d_weights,
                          d_grad_dec2_up, conv5->d_grad_weights, conv5->d_grad_bias,
                          batch_size, 32, 32, 256, 3, 3, 1, 1);
    
    // Backward through Upsample2
    launch_upsample2d_backward(d_grad_dec2_up, d_grad_dec2_relu,
                               batch_size, 16, 16, 256, 2);
    
    // Backward through ReLU (Dec2)
    launch_relu_backward(d_grad_dec2_relu, d_dec2, d_grad_dec2,
                        batch_size * 16 * 16 * 256);
    
    // Backward through Conv4
    launch_conv2d_backward(d_grad_dec2, d_dec1_up, conv4->d_weights,
                          d_grad_dec1_up, conv4->d_grad_weights, conv4->d_grad_bias,
                          batch_size, 16, 16, 128, 256, 3, 1, 1);
    
    // Backward through Upsample1
    launch_upsample2d_backward(d_grad_dec1_up, d_grad_dec1_relu,
                               batch_size, 8, 8, 128, 2);
    
    // Backward through ReLU (Dec1)
    launch_relu_backward(d_grad_dec1_relu, d_dec1, d_grad_dec1,
                        batch_size * 8 * 8 * 128);
    
    // Backward through Conv3
    launch_conv2d_backward(d_grad_dec1, d_latent, conv3->d_weights,
                          d_grad_latent, conv3->d_grad_weights, conv3->d_grad_bias,
                          batch_size, 8, 8, 128, 128, 3, 1, 1);
    
    // ========== ENCODER BACKWARD ==========
    
    // Backward through MaxPool2
    launch_maxpool2d_backward(d_grad_latent, d_enc2_relu, d_pool2_indices,
                             d_latent, d_grad_enc2_relu,
                             batch_size, 16, 16, 128, 2, 2);
    
    // Backward through ReLU (Enc2)
    launch_relu_backward(d_grad_enc2_relu, d_enc2, d_grad_enc2,
                        batch_size * 16 * 16 * 128);
    
    // Backward through Conv2
    launch_conv2d_backward(d_grad_enc2, d_enc1_pool, conv2->d_weights,
                          d_grad_enc1_pool, conv2->d_grad_weights, conv2->d_grad_bias,
                          batch_size, 16, 16, 256, 128, 3, 1, 1);
    
    // Backward through MaxPool1
    launch_maxpool2d_backward(d_grad_enc1_pool, d_enc1_relu, d_pool1_indices,
                             d_enc1_pool, d_grad_enc1_relu,
                             batch_size, 32, 32, 256, 2, 2);
    
    // Backward through ReLU (Enc1)
    launch_relu_backward(d_grad_enc1_relu, d_enc1, d_grad_enc1,
                        batch_size * 32 * 32 * 256);
    
    // Backward through Conv1
    float *d_grad_input;
    CUDA_CHECK(cudaMalloc(&d_grad_input, batch_size * 32 * 32 * 3 * sizeof(float)));
    launch_conv2d_backward(d_grad_enc1, d_input, conv1->d_weights,
                          d_grad_input, conv1->d_grad_weights, conv1->d_grad_bias,
                          batch_size, 32, 32, 3, 256, 3, 1, 1);
    cudaFree(d_grad_input);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Autoencoder::update_weights() {
    // Update all layer weights using SGD
    launch_sgd_update(conv1->d_weights, conv1->d_grad_weights, 
                     learning_rate, conv1->weight_size);
    launch_sgd_update(conv1->d_bias, conv1->d_grad_bias, 
                     learning_rate, conv1->bias_size);
    
    launch_sgd_update(conv2->d_weights, conv2->d_grad_weights, 
                     learning_rate, conv2->weight_size);
    launch_sgd_update(conv2->d_bias, conv2->d_grad_bias, 
                     learning_rate, conv2->bias_size);
    
    launch_sgd_update(conv3->d_weights, conv3->d_grad_weights, 
                     learning_rate, conv3->weight_size);
    launch_sgd_update(conv3->d_bias, conv3->d_grad_bias, 
                     learning_rate, conv3->bias_size);
    
    launch_sgd_update(conv4->d_weights, conv4->d_grad_weights, 
                     learning_rate, conv4->weight_size);
    launch_sgd_update(conv4->d_bias, conv4->d_grad_bias, 
                     learning_rate, conv4->bias_size);
    
    launch_sgd_update(conv5->d_weights, conv5->d_grad_weights, 
                     learning_rate, conv5->weight_size);
    launch_sgd_update(conv5->d_bias, conv5->d_grad_bias, 
                     learning_rate, conv5->bias_size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Autoencoder::save_weights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving weights: " << filename << std::endl;
        return;
    }
    
    auto save_layer = [&](ConvLayer* layer) {
        std::vector<float> h_weights(layer->weight_size);
        std::vector<float> h_bias(layer->bias_size);
        
        CUDA_CHECK(cudaMemcpy(h_weights.data(), layer->d_weights,
                             layer->weight_size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bias.data(), layer->d_bias,
                             layer->bias_size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        file.write(reinterpret_cast<char*>(h_weights.data()), 
                  layer->weight_size * sizeof(float));
        file.write(reinterpret_cast<char*>(h_bias.data()), 
                  layer->bias_size * sizeof(float));
    };
    
    save_layer(conv1);
    save_layer(conv2);
    save_layer(conv3);
    save_layer(conv4);
    save_layer(conv5);
    
    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

void Autoencoder::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading weights: " << filename << std::endl;
        return;
    }
    
    auto load_layer = [&](ConvLayer* layer) {
        std::vector<float> h_weights(layer->weight_size);
        std::vector<float> h_bias(layer->bias_size);
        
        file.read(reinterpret_cast<char*>(h_weights.data()), 
                 layer->weight_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_bias.data()), 
                 layer->bias_size * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(layer->d_weights, h_weights.data(),
                             layer->weight_size * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layer->d_bias, h_bias.data(),
                             layer->bias_size * sizeof(float), 
                             cudaMemcpyHostToDevice));
    };
    
    load_layer(conv1);
    load_layer(conv2);
    load_layer(conv3);
    load_layer(conv4);
    load_layer(conv5);
    
    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
}

void Autoencoder::get_output(float* h_output) {
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 
                         batch_size * 32 * 32 * 3 * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

// ============================================================================
// Training Function
// ============================================================================

void Autoencoder::train(const std::vector<float>& train_data, 
                      int num_samples, int epochs) {
    
    int num_batches = num_samples / batch_size;
    
    // Allocate device memory for batch
    float *d_batch_input, *d_batch_target;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_size * 32 * 32 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_target, batch_size * 32 * 32 * 3 * sizeof(float)));
        
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // GPU timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        for (int batch = 0; batch < num_batches; batch++) {
            int offset = batch * batch_size * 32 * 32 * 3;
            
            // Copy batch to device
            CUDA_CHECK(cudaMemcpy(d_batch_input, 
                                 train_data.data() + offset,
                                 batch_size * 32 * 32 * 3 * sizeof(float),
                                 cudaMemcpyHostToDevice));
            
            // Target is same as input for autoencoder
            CUDA_CHECK(cudaMemcpy(d_batch_target, 
                                 train_data.data() + offset,
                                 batch_size * 32 * 32 * 3 * sizeof(float),
                                 cudaMemcpyHostToDevice));
            
            // Forward pass
            float loss = forward(d_batch_input, d_batch_target);
            epoch_loss += loss;
            
            // Backward pass
            backward(d_batch_input, d_batch_target);
            
            // Update weights
            update_weights();
            
            // Display progress
            if (batch % 10 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
                         << "Batch [" << batch << "/" << num_batches << "] "
                         << "Loss: " << loss << std::endl;
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        epoch_loss /= num_batches;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Epoch " << epoch + 1 << " Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Average Loss: " << epoch_loss << std::endl;
        std::cout << "Time: " << milliseconds / 1000.0f << " seconds" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Save trained weights
    save_weights("autoencoder_weights.bin");
    std::cout << "\nTraining completed!" << std::endl;
    
    cudaFree(d_batch_input);
    cudaFree(d_batch_target);
}
