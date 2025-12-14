#ifndef AUTOENCODER_GPU_H
#define AUTOENCODER_GPU_H

#include <vector>
#include <string>

class AutoencoderGPU {
public:
    AutoencoderGPU();
    ~AutoencoderGPU();

    // Train one step - input_chw: float[3072] in CHW format, normalized [0,1]
    float train_step(const float* input_chw, float learning_rate);
    
    // Train on full dataset
    void train(const std::vector<float>& train_images,
               int num_train_images,
               int batch_size,
               int epochs,
               float learning_rate);
    
    // Save/Load weights
    bool save_weights(const std::string& filepath) const;
    bool load_weights(const std::string& filepath);
    
    // Get last loss
    float get_loss() const;
    
    // Extract features from encoder (bottleneck: 128*8*8 = 8192 features)
    void extract_features(const float* input_chw, float* output_features);

private:
    // Network architecture parameters
    static constexpr int INPUT_H = 32;
    static constexpr int INPUT_W = 32;
    static constexpr int INPUT_C = 3;
    
    static constexpr int CONV1_FILTERS = 256;
    static constexpr int CONV2_FILTERS = 128;
    static constexpr int LATENT_H = 8;
    static constexpr int LATENT_W = 8;
    static constexpr int LATENT_C = 128;
    static constexpr int LATENT_DIM = LATENT_H * LATENT_W * LATENT_C;
    
    // Host weight storage (mutable for save_weights const method)
    mutable std::vector<float> h_conv1_weight;
    mutable std::vector<float> h_conv1_bias;
    mutable std::vector<float> h_conv2_weight;
    mutable std::vector<float> h_conv2_bias;
    mutable std::vector<float> h_conv3_weight;
    mutable std::vector<float> h_conv3_bias;
    mutable std::vector<float> h_conv4_weight;
    mutable std::vector<float> h_conv4_bias;
    mutable std::vector<float> h_conv5_weight;
    mutable std::vector<float> h_conv5_bias;
    
    // Device weight pointers
    float* d_conv1_weight;
    float* d_conv1_bias;
    float* d_conv2_weight;
    float* d_conv2_bias;
    float* d_conv3_weight;
    float* d_conv3_bias;
    float* d_conv4_weight;
    float* d_conv4_bias;
    float* d_conv5_weight;
    float* d_conv5_bias;
    
    // Device gradient pointers
    float* d_conv1_weight_grad;
    float* d_conv1_bias_grad;
    float* d_conv2_weight_grad;
    float* d_conv2_bias_grad;
    float* d_conv3_weight_grad;
    float* d_conv3_bias_grad;
    float* d_conv4_weight_grad;
    float* d_conv4_bias_grad;
    float* d_conv5_weight_grad;
    float* d_conv5_bias_grad;
    
    // Device activation buffers
    float* d_input;
    float* d_conv1_out;
    float* d_relu1_out;
    float* d_pool1_out;
    float* d_conv2_out;
    float* d_relu2_out;
    float* d_pool2_out;
    float* d_conv3_out;
    float* d_relu3_out;
    float* d_up1_out;
    float* d_conv4_out;
    float* d_relu4_out;
    float* d_up2_out;
    float* d_conv5_out;
    
    // Device gradient buffers
    float* d_grad_conv5;
    float* d_grad_up2;
    float* d_grad_relu4;
    float* d_grad_conv4;
    float* d_grad_up1;
    float* d_grad_relu3;
    float* d_grad_conv3;
    float* d_grad_pool2;
    float* d_grad_relu2;
    float* d_grad_conv2;
    float* d_grad_pool1;
    float* d_grad_relu1;
    float* d_grad_conv1;
    
    float* d_loss;
    float last_loss;
    
    void forward();
    void backward();
    void update_weights(float learning_rate);
};

#endif // AUTOENCODER_GPU_H
