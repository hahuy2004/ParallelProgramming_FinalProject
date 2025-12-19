#include "../include/autoencoder_cpu.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>

// Constructor: Khởi tạo autoencoder
AutoencoderCPU::AutoencoderCPU() : current_batch_size(0) {
    initialize_weights();
}

AutoencoderCPU::~AutoencoderCPU() {
}

// Khởi tạo weights bằng Xavier initialization
void AutoencoderCPU::initialize_weights() {
    // Dùng seed cố định để reproducible (có thể uncomment random_device để random)
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);      // Fixed seed
    
    // Xavier/He initialization
    auto init_conv = [&](std::vector<float>& weights, int in_c, int out_c, int k) {
        int fan_in = in_c * k * k;
        float std = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        weights.resize(out_c * in_c * k * k);
        for (auto& w : weights) w = dist(gen);
    };
    
    // Initialize all conv layers
    init_conv(conv1_weight, INPUT_C, CONV1_FILTERS, 3);
    conv1_bias.resize(CONV1_FILTERS, 0.0f);
    
    init_conv(conv2_weight, CONV1_FILTERS, CONV2_FILTERS, 3);
    conv2_bias.resize(CONV2_FILTERS, 0.0f);
    
    init_conv(conv3_weight, LATENT_C, LATENT_C, 3);
    conv3_bias.resize(LATENT_C, 0.0f);
    
    init_conv(conv4_weight, LATENT_C, CONV1_FILTERS, 3);
    conv4_bias.resize(CONV1_FILTERS, 0.0f);
    
    init_conv(conv5_weight, CONV1_FILTERS, INPUT_C, 3);
    conv5_bias.resize(INPUT_C, 0.0f);
}

void AutoencoderCPU::allocate_buffers(int batch_size) {
    if (batch_size == current_batch_size) return;
    current_batch_size = batch_size;
    
    // Allocate activation buffers
    conv1_out.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    pool1_out.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    conv2_out.resize(batch_size * 16 * 16 * CONV2_FILTERS);
    pool2_out.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    conv3_out.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    up1_out.resize(batch_size * 16 * 16 * LATENT_C);
    conv4_out.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    up2_out.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    conv5_out.resize(batch_size * INPUT_H * INPUT_W * INPUT_C);
    
    // Allocate gradient buffers
    grad_conv5.resize(batch_size * INPUT_H * INPUT_W * INPUT_C);
    grad_up2.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    grad_conv4.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    grad_up1.resize(batch_size * 16 * 16 * LATENT_C);
    grad_conv3.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    grad_pool2.resize(batch_size * LATENT_H * LATENT_W * LATENT_C);
    grad_conv2.resize(batch_size * 16 * 16 * CONV2_FILTERS);
    grad_pool1.resize(batch_size * 16 * 16 * CONV1_FILTERS);
    grad_conv1.resize(batch_size * INPUT_H * INPUT_W * CONV1_FILTERS);
    
    conv1_weight_grad.resize(conv1_weight.size(), 0.0f);
    conv1_bias_grad.resize(conv1_bias.size(), 0.0f);
    conv2_weight_grad.resize(conv2_weight.size(), 0.0f);
    conv2_bias_grad.resize(conv2_bias.size(), 0.0f);
    conv3_weight_grad.resize(conv3_weight.size(), 0.0f);
    conv3_bias_grad.resize(conv3_bias.size(), 0.0f);
    conv4_weight_grad.resize(conv4_weight.size(), 0.0f);
    conv4_bias_grad.resize(conv4_bias.size(), 0.0f);
    conv5_weight_grad.resize(conv5_weight.size(), 0.0f);
    conv5_bias_grad.resize(conv5_bias.size(), 0.0f);
}

// Convolution 2D forward pass
// Tính output = conv(input, weights) + bias
void AutoencoderCPU::conv2d_forward(const float* input, float* output,
                                     const float* weights, const float* bias,
                                     int batch, int in_h, int in_w, int in_c,
                                     int out_c, int kernel_size, int stride, int padding) {
    // Tính kích thước output
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
                                    // CHW layout: [batch][channel][height][width]
                                    int in_idx = b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw;
                                    int w_idx = oc * in_c * kernel_size * kernel_size + 
                                               ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    sum += input[in_idx] * weights[w_idx];
                                }
                            }
                        }
                    }
                    
                    // CHW layout: [batch][channel][height][width]
                    int out_idx = b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ReLU activation: max(0, x)
void AutoencoderCPU::relu_forward(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// Max pooling 2D: Lấy giá trị lớn nhất trong mỗi pool window
void AutoencoderCPU::maxpool2d_forward(const float* input, float* output,
                                        int batch, int h, int w, int c,
                                        int pool_size, int stride) {
    // Tính kích thước output sau pooling
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
                            // CHW layout: [batch][channel][height][width]
                            int in_idx = b * c * h * w + ch * h * w + ih * w + iw;
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }
                    
                    // CHW layout
                    int out_idx = b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

// Upsampling 2D: Phóng to ảnh bằng nearest neighbor interpolation
void AutoencoderCPU::upsample2d_forward(const float* input, float* output,
                                         int batch, int in_h, int in_w, int c,
                                         int scale_factor) {
    // Kích thước output sau khi phóng to
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    // CHW layout: [batch][channel][height][width]
                    int in_idx = b * c * in_h * in_w + ch * in_h * in_w + ih * in_w + iw;
                    int out_idx = b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// Convolution 2D backward pass
// Tính gradient cho input, weights và bias từ gradient của output
void AutoencoderCPU::conv2d_backward(const float* grad_output, float* grad_input,
                                      float* grad_weights, float* grad_bias,
                                      const float* input, const float* weights,
                                      int batch, int in_h, int in_w, int in_c,
                                      int out_c, int kernel_size, int stride, int padding) {
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Khởi tạo gradients = 0
    if (grad_input) {
        std::fill(grad_input, grad_input + batch * in_h * in_w * in_c, 0.0f);
    }
    
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // CHW layout: [batch][channel][height][width]
                    int out_idx = b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                    float grad_out_val = grad_output[out_idx];
                    
                    // Gradient w.r.t bias
                    grad_bias[oc] += grad_out_val;
                    
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    // CHW layout: [batch][channel][height][width]
                                    int in_idx = b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw;
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

// ReLU backward: gradient = 0 nếu output <= 0, giữ nguyên nếu output > 0
void AutoencoderCPU::relu_backward(const float* grad_output, float* grad_input,
                                    const float* output, int size) {
    for (int i = 0; i < size; ++i) {
        grad_input[i] = (output[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

// Max pooling backward: Gradient chỉ truyền về vị trí có giá trị max
void AutoencoderCPU::maxpool2d_backward(const float* grad_output, float* grad_input,
                                         const float* input, const float* output,
                                         int batch, int h, int w, int c,
                                         int pool_size, int stride) {
    int out_h = (h - pool_size) / stride + 1;
    int out_w = (w - pool_size) / stride + 1;
    
    // Khởi tạo gradient input = 0
    std::fill(grad_input, grad_input + batch * h * w * c, 0.0f);
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // CHW layout: [batch][channel][height][width]
                    int out_idx = b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                    float max_val = output[out_idx];
                    float grad_val = grad_output[out_idx];
                    
                    // Find which input element was the max and assign gradient to it
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            // CHW layout
                            int in_idx = b * c * h * w + ch * h * w + ih * w + iw;
                            
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

// Upsample backward: Cộng dồn gradient từ các vị trí được duplicate
void AutoencoderCPU::upsample2d_backward(const float* grad_output, float* grad_input,
                                          int batch, int in_h, int in_w, int c,
                                          int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    
    // Khởi tạo gradient input = 0
    std::fill(grad_input, grad_input + batch * in_h * in_w * c, 0.0f);
    
    for (int b = 0; b < batch; ++b) {
        for (int ch = 0; ch < c; ++ch) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    // CHW layout: [batch][channel][height][width]
                    int in_idx = b * c * in_h * in_w + ch * in_h * in_w + ih * in_w + iw;
                    int out_idx = b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow;
                    
                    // Sum all gradients that correspond to the same input pixel
                    grad_input[in_idx] += grad_output[out_idx];
                }
            }
        }
    }
}

// Forward pass: Chạy toàn bộ mạng từ input đến output
void AutoencoderCPU::forward(const float* input, int batch_size) {
    allocate_buffers(batch_size);
    
    // ===== ENCODER: Nén ảnh từ 32x32x3 -> 8x8x128 =====
    // Conv1 + ReLU: (32, 32, 3) -> (32, 32, 256)
    conv2d_forward(input, conv1_out.data(), conv1_weight.data(), conv1_bias.data(),
                   batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
    relu_forward(conv1_out.data(), conv1_out.data(), conv1_out.size());
    
    // MaxPool: (32, 32, 256) -> (16, 16, 256)
    maxpool2d_forward(conv1_out.data(), pool1_out.data(),
                      batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
    
    // Conv2 + ReLU: (16, 16, 256) -> (16, 16, 128)
    conv2d_forward(pool1_out.data(), conv2_out.data(), conv2_weight.data(), conv2_bias.data(),
                   batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
    relu_forward(conv2_out.data(), conv2_out.data(), conv2_out.size());
    
    // MaxPool: (16, 16, 128) -> (8, 8, 128) - Latent representation
    maxpool2d_forward(conv2_out.data(), pool2_out.data(),
                      batch_size, 16, 16, CONV2_FILTERS, 2, 2);
    
    // ===== DECODER: Giải nén từ 8x8x128 -> 32x32x3 =====
    // Conv3 + ReLU: (8, 8, 128) -> (8, 8, 128)
    conv2d_forward(pool2_out.data(), conv3_out.data(), conv3_weight.data(), conv3_bias.data(),
                   batch_size, LATENT_H, LATENT_W, LATENT_C, LATENT_C, 3, 1, 1);
    relu_forward(conv3_out.data(), conv3_out.data(), conv3_out.size());
    
    // Upsample: (8, 8, 128) -> (16, 16, 128)
    upsample2d_forward(conv3_out.data(), up1_out.data(),
                       batch_size, LATENT_H, LATENT_W, LATENT_C, 2);
    
    // Conv4 + ReLU: (16, 16, 128) -> (16, 16, 256)
    conv2d_forward(up1_out.data(), conv4_out.data(), conv4_weight.data(), conv4_bias.data(),
                   batch_size, 16, 16, LATENT_C, CONV1_FILTERS, 3, 1, 1);
    relu_forward(conv4_out.data(), conv4_out.data(), conv4_out.size());
    
    // Upsample: (16, 16, 256) -> (32, 32, 256)
    upsample2d_forward(conv4_out.data(), up2_out.data(),
                       batch_size, 16, 16, CONV1_FILTERS, 2);
    
    // Conv5 (no activation): (32, 32, 256) -> (32, 32, 3)
    conv2d_forward(up2_out.data(), conv5_out.data(), conv5_weight.data(), conv5_bias.data(),
                   batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, INPUT_C, 3, 1, 1);
}

// Backward pass: Tính gradient và update tất cả weights
void AutoencoderCPU::backward(const float* input, int batch_size) {
    // Tính gradient của loss (MSE): d_loss/d_output = 2 * (output - target) / N
    int total_elements = grad_conv5.size();
    for (size_t i = 0; i < grad_conv5.size(); ++i) {
        grad_conv5[i] = 2.0f * (conv5_out[i] - input[i]) / total_elements;
    }
    
    // Clear all weight gradients
    std::fill(conv1_weight_grad.begin(), conv1_weight_grad.end(), 0.0f);
    std::fill(conv1_bias_grad.begin(), conv1_bias_grad.end(), 0.0f);
    std::fill(conv2_weight_grad.begin(), conv2_weight_grad.end(), 0.0f);
    std::fill(conv2_bias_grad.begin(), conv2_bias_grad.end(), 0.0f);
    std::fill(conv3_weight_grad.begin(), conv3_weight_grad.end(), 0.0f);
    std::fill(conv3_bias_grad.begin(), conv3_bias_grad.end(), 0.0f);
    std::fill(conv4_weight_grad.begin(), conv4_weight_grad.end(), 0.0f);
    std::fill(conv4_bias_grad.begin(), conv4_bias_grad.end(), 0.0f);
    std::fill(conv5_weight_grad.begin(), conv5_weight_grad.end(), 0.0f);
    std::fill(conv5_bias_grad.begin(), conv5_bias_grad.end(), 0.0f);
    
    // ===== DECODER BACKWARD: Tính gradient từ output về latent =====
    // Backward qua Conv5: (32, 32, 256) -> (32, 32, 3)
    conv2d_backward(grad_conv5.data(), grad_up2.data(),
                    conv5_weight_grad.data(), conv5_bias_grad.data(),
                    up2_out.data(), conv5_weight.data(),
                    batch_size, INPUT_H, INPUT_W, CONV1_FILTERS,
                    INPUT_C, 3, 1, 1);
    
    // Backward through UpSample2: (16, 16, 256) -> (32, 32, 256)
    upsample2d_backward(grad_up2.data(), grad_conv4.data(),
                        batch_size, 16, 16, CONV1_FILTERS, 2);
    
    // Backward through ReLU for Conv4
    relu_backward(grad_conv4.data(), grad_conv4.data(),
                  conv4_out.data(), grad_conv4.size());
    
    // Backward through Conv4: (16, 16, 128) -> (16, 16, 256)
    conv2d_backward(grad_conv4.data(), grad_up1.data(),
                    conv4_weight_grad.data(), conv4_bias_grad.data(),
                    up1_out.data(), conv4_weight.data(),
                    batch_size, 16, 16, LATENT_C,
                    CONV1_FILTERS, 3, 1, 1);
    
    // Backward through UpSample1: (8, 8, 128) -> (16, 16, 128)
    upsample2d_backward(grad_up1.data(), grad_conv3.data(),
                        batch_size, LATENT_H, LATENT_W, LATENT_C, 2);
    
    // Backward through ReLU for Conv3
    relu_backward(grad_conv3.data(), grad_conv3.data(),
                  conv3_out.data(), grad_conv3.size());
    
    // Backward through Conv3: (8, 8, 128) -> (8, 8, 128)
    conv2d_backward(grad_conv3.data(), grad_pool2.data(),
                    conv3_weight_grad.data(), conv3_bias_grad.data(),
                    pool2_out.data(), conv3_weight.data(),
                    batch_size, LATENT_H, LATENT_W, LATENT_C,
                    LATENT_C, 3, 1, 1);
    
    // ===== ENCODER BACKWARD: Tính gradient từ latent về input =====
    // Backward qua MaxPool2: (16, 16, 128) -> (8, 8, 128)
    maxpool2d_backward(grad_pool2.data(), grad_conv2.data(),
                       conv2_out.data(), pool2_out.data(),
                       batch_size, 16, 16, CONV2_FILTERS, 2, 2);
    
    // Backward through ReLU for Conv2
    relu_backward(grad_conv2.data(), grad_conv2.data(),
                  conv2_out.data(), grad_conv2.size());
    
    // Backward through Conv2: (16, 16, 256) -> (16, 16, 128)
    conv2d_backward(grad_conv2.data(), grad_pool1.data(),
                    conv2_weight_grad.data(), conv2_bias_grad.data(),
                    pool1_out.data(), conv2_weight.data(),
                    batch_size, 16, 16, CONV1_FILTERS,
                    CONV2_FILTERS, 3, 1, 1);
    
    // Backward through MaxPool1: (32, 32, 256) -> (16, 16, 256)
    maxpool2d_backward(grad_pool1.data(), grad_conv1.data(),
                       conv1_out.data(), pool1_out.data(),
                       batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
    
    // Backward through ReLU for Conv1
    relu_backward(grad_conv1.data(), grad_conv1.data(),
                  conv1_out.data(), grad_conv1.size());
    
    // Backward through Conv1: (32, 32, 3) -> (32, 32, 256)
    // Don't need grad_input for the first layer
    std::vector<float> temp_input(batch_size * INPUT_H * INPUT_W * INPUT_C);
    std::memcpy(temp_input.data(), input, temp_input.size() * sizeof(float));
    
    conv2d_backward(grad_conv1.data(), nullptr,
                    conv1_weight_grad.data(), conv1_bias_grad.data(),
                    temp_input.data(), conv1_weight.data(),
                    batch_size, INPUT_H, INPUT_W, INPUT_C,
                    CONV1_FILTERS, 3, 1, 1);
}

// Update weights: Gradient descent đơn giản (weight -= lr * gradient)
void AutoencoderCPU::update_weights(float learning_rate) {
    // Update weights của tất cả các layer
    for (size_t i = 0; i < conv1_weight.size(); ++i) {
        conv1_weight[i] -= learning_rate * conv1_weight_grad[i];
    }
    for (size_t i = 0; i < conv2_weight.size(); ++i) {
        conv2_weight[i] -= learning_rate * conv2_weight_grad[i];
    }
    for (size_t i = 0; i < conv3_weight.size(); ++i) {
        conv3_weight[i] -= learning_rate * conv3_weight_grad[i];
    }
    for (size_t i = 0; i < conv4_weight.size(); ++i) {
        conv4_weight[i] -= learning_rate * conv4_weight_grad[i];
    }
    for (size_t i = 0; i < conv5_weight.size(); ++i) {
        conv5_weight[i] -= learning_rate * conv5_weight_grad[i];
    }
    
    // Simple gradient descent for biases
    for (size_t i = 0; i < conv1_bias.size(); ++i) {
        conv1_bias[i] -= learning_rate * conv1_bias_grad[i];
    }
    for (size_t i = 0; i < conv2_bias.size(); ++i) {
        conv2_bias[i] -= learning_rate * conv2_bias_grad[i];
    }
    for (size_t i = 0; i < conv3_bias.size(); ++i) {
        conv3_bias[i] -= learning_rate * conv3_bias_grad[i];
    }
    for (size_t i = 0; i < conv4_bias.size(); ++i) {
        conv4_bias[i] -= learning_rate * conv4_bias_grad[i];
    }
    for (size_t i = 0; i < conv5_bias.size(); ++i) {
        conv5_bias[i] -= learning_rate * conv5_bias_grad[i];
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
                conv5_out, actual_batch_size);
            
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
        
        conv2d_forward(batch_data, conv1_out.data(), conv1_weight.data(), conv1_bias.data(),
                       actual_batch_size, INPUT_H, INPUT_W, INPUT_C, CONV1_FILTERS, 3, 1, 1);
        relu_forward(conv1_out.data(), conv1_out.data(), conv1_out.size());
        
        maxpool2d_forward(conv1_out.data(), pool1_out.data(),
                          actual_batch_size, INPUT_H, INPUT_W, CONV1_FILTERS, 2, 2);
        
        conv2d_forward(pool1_out.data(), conv2_out.data(), conv2_weight.data(), conv2_bias.data(),
                       actual_batch_size, 16, 16, CONV1_FILTERS, CONV2_FILTERS, 3, 1, 1);
        relu_forward(conv2_out.data(), conv2_out.data(), conv2_out.size());
        
        maxpool2d_forward(conv2_out.data(), pool2_out.data(),
                          actual_batch_size, 16, 16, CONV2_FILTERS, 2, 2);
        
        // Copy latent features
        std::memcpy(&features[start_idx * LATENT_DIM],
                   pool2_out.data(),
                   actual_batch_size * LATENT_DIM * sizeof(float));
    }
}

void AutoencoderCPU::reconstruct(const float* input_chw, float* output_reconstructed) {
    const int image_size = INPUT_C * INPUT_H * INPUT_W; // 3072 pixels
    
    // Run full forward pass (encoder + decoder) with batch_size=1
    forward(input_chw, 1);
    
    // Copy reconstructed output
    std::memcpy(output_reconstructed, conv5_out.data(), image_size * sizeof(float));
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
    
    write_vec(conv1_weight);
    write_vec(conv1_bias);
    write_vec(conv2_weight);
    write_vec(conv2_bias);
    write_vec(conv3_weight);
    write_vec(conv3_bias);
    write_vec(conv4_weight);
    write_vec(conv4_bias);
    write_vec(conv5_weight);
    write_vec(conv5_bias);
    
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
    
    read_vec(conv1_weight);
    read_vec(conv1_bias);
    read_vec(conv2_weight);
    read_vec(conv2_bias);
    read_vec(conv3_weight);
    read_vec(conv3_bias);
    read_vec(conv4_weight);
    read_vec(conv4_bias);
    read_vec(conv5_weight);
    read_vec(conv5_bias);
    
    file.close();
    std::cout << "Weights loaded from " << filepath << std::endl;
}

