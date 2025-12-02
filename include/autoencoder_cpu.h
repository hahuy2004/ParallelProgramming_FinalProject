#ifndef AUTOENCODER_CPU_H
#define AUTOENCODER_CPU_H

#include <vector>
#include <string>

class AutoencoderCPU {
public:
    AutoencoderCPU();
    ~AutoencoderCPU();

    // Training
    void train(const std::vector<float>& train_images, 
               int num_images,
               int batch_size, 
               int epochs, 
               float learning_rate);
    
    // Feature extraction (encoder only)
    void extract_features(const std::vector<float>& images,
                         int num_images,
                         std::vector<float>& features);
    
    // Save/Load weights
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);
    
    // Get reconstruction loss
    float compute_loss(const std::vector<float>& input,
                      const std::vector<float>& output,
                      int batch_size);

private:
    // Network architecture parameters
    static constexpr int INPUT_H = 32;
    static constexpr int INPUT_W = 32;
    static constexpr int INPUT_C = 3;
    
    // Layer dimensions
    static constexpr int CONV1_FILTERS = 256;
    static constexpr int CONV2_FILTERS = 128;
    static constexpr int LATENT_H = 8;
    static constexpr int LATENT_W = 8;
    static constexpr int LATENT_C = 128;
    static constexpr int LATENT_DIM = LATENT_H * LATENT_W * LATENT_C;  // 8192
    
    // Weights and biases
    std::vector<float> conv1_weights_;  // [256, 3, 3, 3]
    std::vector<float> conv1_bias_;     // [256]
    std::vector<float> conv2_weights_;  // [128, 256, 3, 3]
    std::vector<float> conv2_bias_;     // [128]
    std::vector<float> conv3_weights_;  // [128, 128, 3, 3]
    std::vector<float> conv3_bias_;     // [128]
    std::vector<float> conv4_weights_;  // [256, 128, 3, 3]
    std::vector<float> conv4_bias_;     // [256]
    std::vector<float> conv5_weights_;  // [3, 256, 3, 3]
    std::vector<float> conv5_bias_;     // [3]
    
    // Activation buffers (for one batch)
    std::vector<float> conv1_out_;
    std::vector<float> pool1_out_;
    std::vector<float> conv2_out_;
    std::vector<float> pool2_out_;  // Latent representation
    std::vector<float> conv3_out_;
    std::vector<float> up1_out_;
    std::vector<float> conv4_out_;
    std::vector<float> up2_out_;
    std::vector<float> conv5_out_;  // Reconstruction
    
    // Gradient buffers
    std::vector<float> grad_conv5_out_;
    std::vector<float> grad_up2_out_;
    std::vector<float> grad_conv4_out_;
    std::vector<float> grad_up1_out_;
    std::vector<float> grad_conv3_out_;
    std::vector<float> grad_pool2_out_;
    std::vector<float> grad_conv2_out_;
    std::vector<float> grad_pool1_out_;
    std::vector<float> grad_conv1_out_;
    
    std::vector<float> grad_conv1_weights_;
    std::vector<float> grad_conv1_bias_;
    std::vector<float> grad_conv2_weights_;
    std::vector<float> grad_conv2_bias_;
    std::vector<float> grad_conv3_weights_;
    std::vector<float> grad_conv3_bias_;
    std::vector<float> grad_conv4_weights_;
    std::vector<float> grad_conv4_bias_;
    std::vector<float> grad_conv5_weights_;
    std::vector<float> grad_conv5_bias_;
    
    int current_batch_size_;
    
    // Layer operations
    void initialize_weights();
    void allocate_buffers(int batch_size);
    
    // Forward pass layers
    void conv2d_forward(const float* input, float* output,
                       const float* weights, const float* bias,
                       int batch, int in_h, int in_w, int in_c,
                       int out_c, int kernel_size, int stride, int padding);
    
    void relu_forward(const float* input, float* output, int size);
    
    void maxpool2d_forward(const float* input, float* output,
                          int batch, int h, int w, int c,
                          int pool_size, int stride);
    
    void upsample2d_forward(const float* input, float* output,
                           int batch, int in_h, int in_w, int c,
                           int scale_factor);
    
    // Backward pass layers
    void conv2d_backward(const float* input, const float* grad_output,
                        float* grad_input, float* grad_weights, float* grad_bias,
                        const float* weights,
                        int batch, int in_h, int in_w, int in_c,
                        int out_c, int kernel_size, int stride, int padding);
    
    void relu_backward(const float* input, const float* grad_output,
                      float* grad_input, int size);
    
    void maxpool2d_backward(const float* input, const float* output,
                           const float* grad_output, float* grad_input,
                           int batch, int h, int w, int c,
                           int pool_size, int stride);
    
    void upsample2d_backward(const float* grad_output, float* grad_input,
                            int batch, int in_h, int in_w, int c,
                            int scale_factor);
    
    // Full forward and backward
    void forward(const float* input, int batch_size);
    void backward(const float* input, int batch_size);
    void update_weights(float learning_rate);
};

#endif // AUTOENCODER_CPU_H
