#include "../include/autoencoder_cpu.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>

AutoencoderCPU::AutoencoderCPU() : current_batch_size_(0) {
    initialize_weights();
}

AutoencoderCPU::~AutoencoderCPU() {
}

void AutoencoderCPU::initialize_weights() {
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);      // Fixed seed for reproducibility
    
    // Xavier/He initialization
    auto init_conv = [&](std::vector<float>& weights, int in_c, int out_c, int k) {
        int fan_in = in_c * k * k;
        float std = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        weights.resize(out_c * in_c * k * k);
        for (auto& w : weights) w = dist(gen);
    };
    
    // Initialize all conv layers
    init_conv(conv1_weights_, INPUT_C, CONV1_FILTERS, 3);
    conv1_bias_.resize(CONV1_FILTERS, 0.0f);
    
    init_conv(conv2_weights_, CONV1_FILTERS, CONV2_FILTERS, 3);
    conv2_bias_.resize(CONV2_FILTERS, 0.0f);
    
    init_conv(conv3_weights_, LATENT_C, LATENT_C, 3);
    conv3_bias_.resize(LATENT_C, 0.0f);
    
    init_conv(conv4_weights_, LATENT_C, CONV1_FILTERS, 3);
    conv4_bias_.resize(CONV1_FILTERS, 0.0f);
    
    init_conv(conv5_weights_, CONV1_FILTERS, INPUT_C, 3);
    conv5_bias_.resize(INPUT_C, 0.0f);
}

void AutoencoderCPU::allocate_buffers(int batch_size) {
    if (batch_size == current_batch_size_) return;
    current_batch_size_ = batch_size;
    
    // Allocate activation buffers
    conv1_out_.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    pool1_out_.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    conv2_out_.resize(batch_size * 16 * 16 * CONV2_FILTERS);
    pool2_out_.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    conv3_out_.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    up1_out_.resize(batch_size * 16 * 16 * LATENT_C);
    conv4_out_.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    up2_out_.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    conv5_out_.resize(batch_size * INPUT_H * INPUT_W * INPUT_C);
    
    // Allocate gradient buffers
    grad_conv5_out_.resize(batch_size * INPUT_H * INPUT_W * INPUT_C);
    grad_up2_out_.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    grad_conv4_out_.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    grad_up1_out_.resize(batch_size * 16 * 16 * LATENT_C);
    grad_conv3_out_.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    grad_pool2_out_.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    grad_conv2_out_.resize(batch_size * 16 * 16 * CONV2_FILTERS);
    grad_pool1_out_.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    grad_conv1_out_.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    
    grad_conv1_weights_.resize(conv1_weights_.size(), 0.0f);
    grad_conv1_bias_.resize(conv1_bias_.size(), 0.0f);
    grad_conv2_weights_.resize(conv2_weights_.size(), 0.0f);
    grad_conv2_bias_.resize(conv2_bias_.size(), 0.0f);
    grad_conv3_weights_.resize(conv3_weights_.size(), 0.0f);
    grad_conv3_bias_.resize(conv3_bias_.size(), 0.0f);
    grad_conv4_weights_.resize(conv4_weights_.size(), 0.0f);
    grad_conv4_bias_.resize(conv4_bias_.size(), 0.0f);
    grad_conv5_weights_.resize(conv5_weights_.size(), 0.0f);
    grad_conv5_bias_.resize(conv5_bias_.size(), 0.0f);
}

void AutoencoderCPU::conv2d_forward(const float* input, float* output,
                                     const float* weights, const float* bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
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
                    
                    int out_idx = b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void AutoencoderCPU::relu_forward(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void AutoencoderCPU::maxpool2d_forward(const float* input, float* output,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -INFINITY;
                    
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            int in_idx = b * h * w * c + ih * w * c + iw * c + ch;
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }
                    
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

void AutoencoderCPU::upsample2d_forward(const float* input, float* output,
                                         int batch, int in_h, int in_w, int c,
                                         int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    int in_idx = b * in_h * in_w * c + ih * in_w * c + iw * c + ch;
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void AutoencoderCPU::conv2d_backward(const float* grad_output, float* grad_input,
                                      float* grad_weights, float* grad_bias,
                                      const float* input, const float* weights,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Initialize gradients to zero
    if (grad_input) {
        std::fill(grad_input, grad_input + batch * in_h * in_w * in_c, 0.0f);
    }
    
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = b * out_h * out_w * out_c + oh * out_w * out_c + ow * out_c + oc;
                    float grad_out_val = grad_output[out_idx];
                    
                    // Gradient w.r.t bias
                    grad_bias[oc] += grad_out_val;
                    
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = b * in_h * in_w * in_c + ih * in_w * in_c + iw * in_c + ic;
                                    int w_idx = oc * in_c * kernel_size * kernel_size + 
                                               ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    
                                    // Gradient w.r.t weights
                                    grad_weights[w_idx] += grad_out_val * input[in_idx];
                                    
                                    // Gradient w.r.t input (if needed)
                                    if (grad_input) {
                                        grad_input[in_idx] += grad_out_val * weights[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void AutoencoderCPU::relu_backward(const float* grad_output, float* grad_input,
                                    const float* output, int size) {
    for (int i = 0; i < size; ++i) {
        grad_input[i] = (output[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

void AutoencoderCPU::maxpool2d_backward(const float* grad_output, float* grad_input,
                                         const float* input, const float* output,
                                         int batch, int h, int w, int c,
                                         int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    
    // Initialize gradient input to zero
    std::fill(grad_input, grad_input + batch * h * w * c, 0.0f);
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    float max_val = output[out_idx];
                    float grad_val = grad_output[out_idx];
                    
                    // Find which input element was the max and assign gradient to it
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            int in_idx = b * h * w * c + ih * w * c + iw * c + ch;
                            
                            if (std::abs(input[in_idx] - max_val) < 1e-8f) {
                                grad_input[in_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

void AutoencoderCPU::upsample2d_backward(const float* grad_output, float* grad_input,
                                          int batch, int in_h, int in_w, int c,
                                          int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    // Initialize gradient input to zero
    std::fill(grad_input, grad_input + batch * in_h * in_w * c, 0.0f);
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    int in_idx = b * in_h * in_w * c + ih * in_w * c + iw * c + ch;
                    int out_idx = b * out_h * out_w * c + oh * out_w * c + ow * c + ch;
                    
                    // Sum all gradients that correspond to the same input pixel
                    grad_input[in_idx] += grad_output[out_idx];
                }
            }
        }
    }
}

void AutoencoderCPU::forward(const float* input, int batch_size) {
    allocate_buffers(batch_size);
    
    // ENCODER FORWARD PASS
    // Conv1 + ReLU: (32, 32, 3) -> (32, 32, 256)
    conv2d_forward(input, conv1_out_.data(), conv1_weights_.data(), conv1_bias_.data(),
                   batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
    relu_forward(conv1_out_.data(), conv1_out_.data(), conv1_out_.size());
    
    // MaxPool: (32, 32, 256) -> (16, 16, 256)
    maxpool2d_forward(conv1_out_.data(), pool1_out_.data(),
                      batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
    
    // Conv2 + ReLU: (16, 16, 256) -> (16, 16, 128)
    conv2d_forward(pool1_out_.data(), conv2_out_.data(), conv2_weights_.data(), conv2_bias_.data(),
                   batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
    relu_forward(conv2_out_.data(), conv2_out_.data(), conv2_out_.size());
    
    // MaxPool: (16, 16, 128) -> (8, 8, 128) - Latent representation
    maxpool2d_forward(conv2_out_.data(), pool2_out_.data(),
                      batch_size, 16, 16, CONV2_FILTERS, 2, 2);
    
    // DECODER FORWARD PASS
    // Conv3 + ReLU: (8, 8, 128) -> (8, 8, 128)
    conv2d_forward(pool2_out_.data(), conv3_out_.data(), conv3_weights_.data(), conv3_bias_.data(),
                   batch_size, LATENT_H, LATENT_W, LATENT_C, LATENT_C, 3, 1, 1);
    relu_forward(conv3_out_.data(), conv3_out_.data(), conv3_out_.size());
    
    // Upsample: (8, 8, 128) -> (16, 16, 128)
    upsample2d_forward(conv3_out_.data(), up1_out_.data(),
                       batch_size, LATENT_H, LATENT_W, LATENT_C, 2);
    
    // Conv4 + ReLU: (16, 16, 128) -> (16, 16, 256)
    conv2d_forward(up1_out_.data(), conv4_out_.data(), conv4_weights_.data(), conv4_bias_.data(),
                   batch_size, 16, 16, LATENT_C, CONV1_FILTERS, 3, 1, 1);
    relu_forward(conv4_out_.data(), conv4_out_.data(), conv4_out_.size());
    
    // Upsample: (16, 16, 256) -> (32, 32, 256)
    upsample2d_forward(conv4_out_.data(), up2_out_.data(),
                       batch_size, 16, 16, CONV1_FILTERS, 2);
    
    // Conv5 (no activation): (32, 32, 256) -> (32, 32, 3)
    conv2d_forward(up2_out_.data(), conv5_out_.data(), conv5_weights_.data(), conv5_bias_.data(),
                   batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_C, 3, 1, 1);
}

void AutoencoderCPU::backward(const float* input, int batch_size) {
    // Compute loss gradient: d_loss/d_output = 2 * (output - target) / N
    int total_elements = grad_conv5_out_.size();
    for (size_t i = 0; i < grad_conv5_out_.size(); ++i) {
        grad_conv5_out_[i] = 2.0f * (conv5_out_[i] - input[i]) / total_elements;
    }
    
    // Clear all weight gradients
    std::fill(grad_conv1_weights_.begin(), grad_conv1_weights_.end(), 0.0f);
    std::fill(grad_conv1_bias_.begin(), grad_conv1_bias_.end(), 0.0f);
    std::fill(grad_conv2_weights_.begin(), grad_conv2_weights_.end(), 0.0f);
    std::fill(grad_conv2_bias_.begin(), grad_conv2_bias_.end(), 0.0f);
    std::fill(grad_conv3_weights_.begin(), grad_conv3_weights_.end(), 0.0f);
    std::fill(grad_conv3_bias_.begin(), grad_conv3_bias_.end(), 0.0f);
    std::fill(grad_conv4_weights_.begin(), grad_conv4_weights_.end(), 0.0f);
    std::fill(grad_conv4_bias_.begin(), grad_conv4_bias_.end(), 0.0f);
    std::fill(grad_conv5_weights_.begin(), grad_conv5_weights_.end(), 0.0f);
    std::fill(grad_conv5_bias_.begin(), grad_conv5_bias_.end(), 0.0f);
    
    // DECODER BACKWARD PASS
    // Backward through Conv5: (32, 32, 256) -> (32, 32, 3)
    conv2d_backward(grad_conv5_out_.data(), grad_up2_out_.data(),
                    grad_conv5_weights_.data(), grad_conv5_bias_.data(),
                    up2_out_.data(), conv5_weights_.data(),
                    batch_size, INPUT_H, INPUT_W, CONV1_FILTERS,
                    INPUT_C, 3, 1, 1);
    
    // Backward through UpSample2: (16, 16, 256) -> (32, 32, 256)
    upsample2d_backward(grad_up2_out_.data(), grad_conv4_out_.data(),
                        batch_size, 16, 16, CONV1_FILTERS, 2);
    
    // Backward through ReLU for Conv4
    relu_backward(grad_conv4_out_.data(), grad_conv4_out_.data(),
                  conv4_out_.data(), grad_conv4_out_.size());
    
    // Backward through Conv4: (16, 16, 128) -> (16, 16, 256)
    conv2d_backward(grad_conv4_out_.data(), grad_up1_out_.data(),
                    grad_conv4_weights_.data(), grad_conv4_bias_.data(),
                    up1_out_.data(), conv4_weights_.data(),
                    batch_size, 16, 16, LATENT_C,
                    CONV1_FILTERS, 3, 1, 1);
    
    // Backward through UpSample1: (8, 8, 128) -> (16, 16, 128)
    upsample2d_backward(grad_up1_out_.data(), grad_conv3_out_.data(),
                        batch_size, LATENT_H, LATENT_W, LATENT_C, 2);
    
    // Backward through ReLU for Conv3
    relu_backward(grad_conv3_out_.data(), grad_conv3_out_.data(),
                  conv3_out_.data(), grad_conv3_out_.size());
    
    // Backward through Conv3: (8, 8, 128) -> (8, 8, 128)
    conv2d_backward(grad_conv3_out_.data(), grad_pool2_out_.data(),
                    grad_conv3_weights_.data(), grad_conv3_bias_.data(),
                    pool2_out_.data(), conv3_weights_.data(),
                    batch_size, LATENT_H, LATENT_W, LATENT_C,
                    LATENT_C, 3, 1, 1);
    
    // ENCODER BACKWARD PASS
    // Backward through MaxPool2: (16, 16, 128) -> (8, 8, 128)
    maxpool2d_backward(grad_pool2_out_.data(), grad_conv2_out_.data(),
                       conv2_out_.data(), pool2_out_.data(),
                       batch_size, 16, 16, CONV2_FILTERS, 2, 2);
    
    // Backward through ReLU for Conv2
    relu_backward(grad_conv2_out_.data(), grad_conv2_out_.data(),
                  conv2_out_.data(), grad_conv2_out_.size());
    
    // Backward through Conv2: (16, 16, 256) -> (16, 16, 128)
    conv2d_backward(grad_conv2_out_.data(), grad_pool1_out_.data(),
                    grad_conv2_weights_.data(), grad_conv2_bias_.data(),
                    pool1_out_.data(), conv2_weights_.data(),
                    batch_size, 16, 16, CONV1_FILTERS,
                    CONV2_FILTERS, 3, 1, 1);
    
    // Backward through MaxPool1: (32, 32, 256) -> (16, 16, 256)
    maxpool2d_backward(grad_pool1_out_.data(), grad_conv1_out_.data(),
                       conv1_out_.data(), pool1_out_.data(),
                       batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
    
    // Backward through ReLU for Conv1
    relu_backward(grad_conv1_out_.data(), grad_conv1_out_.data(),
                  conv1_out_.data(), grad_conv1_out_.size());
    
    // Backward through Conv1: (32, 32, 3) -> (32, 32, 256)
    // Don't need grad_input for the first layer
    std::vector<float> temp_input(batch_size * INPUT_H * INPUT_W * INPUT_C);
    std::memcpy(temp_input.data(), input, temp_input.size() * sizeof(float));
    
    conv2d_backward(grad_conv1_out_.data(), nullptr,
                    grad_conv1_weights_.data(), grad_conv1_bias_.data(),
                    temp_input.data(), conv1_weights_.data(),
                    batch_size, INPUT_H, INPUT_W, INPUT_C,
                    CONV1_FILTERS, 3, 1, 1);
}

void AutoencoderCPU::update_weights(float learning_rate) {
    // Simple gradient descent for weights
    for (size_t i = 0; i < conv1_weights_.size(); ++i) {
        conv1_weights_[i] -= learning_rate * grad_conv1_weights_[i];
    }
    for (size_t i = 0; i < conv2_weights_.size(); ++i) {
        conv2_weights_[i] -= learning_rate * grad_conv2_weights_[i];
    }
    for (size_t i = 0; i < conv3_weights_.size(); ++i) {
        conv3_weights_[i] -= learning_rate * grad_conv3_weights_[i];
    }
    for (size_t i = 0; i < conv4_weights_.size(); ++i) {
        conv4_weights_[i] -= learning_rate * grad_conv4_weights_[i];
    }
    for (size_t i = 0; i < conv5_weights_.size(); ++i) {
        conv5_weights_[i] -= learning_rate * grad_conv5_weights_[i];
    }
    
    // Simple gradient descent for biases
    for (size_t i = 0; i < conv1_bias_.size(); ++i) {
        conv1_bias_[i] -= learning_rate * grad_conv1_bias_[i];
    }
    for (size_t i = 0; i < conv2_bias_.size(); ++i) {
        conv2_bias_[i] -= learning_rate * grad_conv2_bias_[i];
    }
    for (size_t i = 0; i < conv3_bias_.size(); ++i) {
        conv3_bias_[i] -= learning_rate * grad_conv3_bias_[i];
    }
    for (size_t i = 0; i < conv4_bias_.size(); ++i) {
        conv4_bias_[i] -= learning_rate * grad_conv4_bias_[i];
    }
    for (size_t i = 0; i < conv5_bias_.size(); ++i) {
        conv5_bias_[i] -= learning_rate * grad_conv5_bias_[i];
    }
}

float AutoencoderCPU::compute_loss(const std::vector<float>& input,
                                    const std::vector<float>& output,
                                    int batch_size) {
    float mse = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float diff = output[i] - input[i];
        mse += diff * diff;
    }
    return mse / input.size();
}

void AutoencoderCPU::train(const std::vector<float>& train_images,
                            int num_images,
                            int batch_size,
                            int epochs,
                            float learning_rate) {
    std::cout << "Training CPU Autoencoder..." << std::endl;
    std::cout << "Images: " << num_images << ", Batch size: " << batch_size 
              << ", Epochs: " << epochs << ", LR: " << learning_rate << std::endl;
    
    int num_batches = (num_images + batch_size - 1) / batch_size;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_images);
            int actual_batch_size = end_idx - start_idx;
            
            const float* batch_data = &train_images[start_idx * INPUT_H * INPUT_W * INPUT_C];
            
            // Forward pass
            forward(batch_data, actual_batch_size);
            
            // Compute loss
            float loss = compute_loss(
                std::vector<float>(batch_data, batch_data + actual_batch_size * INPUT_H * INPUT_W * INPUT_C),
                conv5_out_, actual_batch_size);
            
            // Check for NaN or Inf
            if (std::isnan(loss) || std::isinf(loss)) {
                std::cerr << "\nERROR: Loss is NaN/Inf at epoch " << (epoch + 1) 
                          << ", batch " << (batch + 1) << std::endl;
                std::cerr << "Stopping training..." << std::endl;
                return;
            }
            
            epoch_loss += loss;
            
            // Backward pass
            backward(batch_data, actual_batch_size);
            
            // Update weights
            update_weights(learning_rate);
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " - Loss: " << (epoch_loss / num_batches)
                  << " - Time: " << epoch_time << "s" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(end_time - start_time).count();
    
    std::cout << "Training completed in " << total_time << " seconds" << std::endl;
}

void AutoencoderCPU::extract_features(const std::vector<float>& images,
                                       int num_images,
                                       std::vector<float>& features) {
    features.resize(num_images * LATENT_DIM);
    
    int batch_size = 32;
    int num_batches = (num_images + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_images);
        int actual_batch_size = end_idx - start_idx;
        
        const float* batch_data = &images[start_idx * INPUT_H * INPUT_W * INPUT_C];
        
        // Run encoder only
        allocate_buffers(actual_batch_size);
        
        conv2d_forward(batch_data, conv1_out_.data(), conv1_weights_.data(), conv1_bias_.data(),
                       actual_batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
        relu_forward(conv1_out_.data(), conv1_out_.data(), conv1_out_.size());
        
        maxpool2d_forward(conv1_out_.data(), pool1_out_.data(),
                          actual_batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
        
        conv2d_forward(pool1_out_.data(), conv2_out_.data(), conv2_weights_.data(), conv2_bias_.data(),
                       actual_batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
        relu_forward(conv2_out_.data(), conv2_out_.data(), conv2_out_.size());
        
        maxpool2d_forward(conv2_out_.data(), pool2_out_.data(),
                          actual_batch_size, 16, 16, CONV2_FILTERS, 2, 2);
        
        // Copy latent features
        std::memcpy(&features[start_idx * LATENT_DIM],
                   pool2_out_.data(),
                   actual_batch_size * LATENT_DIM * sizeof(float));
    }
}

void AutoencoderCPU::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving: " << filepath << std::endl;
        return;
    }
    
    auto write_vec = [&file](const std::vector<float>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    };
    
    write_vec(conv1_weights_);
    write_vec(conv1_bias_);
    write_vec(conv2_weights_);
    write_vec(conv2_bias_);
    write_vec(conv3_weights_);
    write_vec(conv3_bias_);
    write_vec(conv4_weights_);
    write_vec(conv4_bias_);
    write_vec(conv5_weights_);
    write_vec(conv5_bias_);
    
    file.close();
    std::cout << "Weights saved to " << filepath << std::endl;
}

void AutoencoderCPU::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading: " << filepath << std::endl;
        return;
    }
    
    auto read_vec = [&file](std::vector<float>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    };
    
    read_vec(conv1_weights_);
    read_vec(conv1_bias_);
    read_vec(conv2_weights_);
    read_vec(conv2_bias_);
    read_vec(conv3_weights_);
    read_vec(conv3_bias_);
    read_vec(conv4_weights_);
    read_vec(conv4_bias_);
    read_vec(conv5_weights_);
    read_vec(conv5_bias_);
    
    file.close();
    std::cout << "Weights loaded from " << filepath << std::endl;
}
