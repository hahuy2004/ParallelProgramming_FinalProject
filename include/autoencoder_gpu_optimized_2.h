#ifndef AUTOENCODER_GPU_OPTIMIZED_2_H
#define AUTOENCODER_GPU_OPTIMIZED_2_H

#include <vector>
#include <string>

class AutoencoderGPUOptimized2 {
public:
    AutoencoderGPUOptimized2();
    ~AutoencoderGPUOptimized2();

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
    void copy_weights_from_cpu(const std::string& cpu_weights_path);

private:
    static constexpr int INPUT_H = 32;
    static constexpr int INPUT_W = 32;
    static constexpr int INPUT_C = 3;
    
    static constexpr int CONV1_FILTERS = 256;
    static constexpr int CONV2_FILTERS = 128;
    static constexpr int LATENT_H = 8;
    static constexpr int LATENT_W = 8;
    static constexpr int LATENT_C = 128;
    static constexpr int LATENT_DIM = LATENT_H * LATENT_W * LATENT_C;
    
    // Device pointers for weights
    float* d_conv1_weights_;
    float* d_conv1_bias_;
    float* d_conv2_weights_;
    float* d_conv2_bias_;
    float* d_conv3_weights_;
    float* d_conv3_bias_;
    float* d_conv4_weights_;
    float* d_conv4_bias_;
    float* d_conv5_weights_;
    float* d_conv5_bias_;
    
    // Device pointers for activations
    float* d_input_;
    float* d_conv1_out_;
    float* d_pool1_out_;
    float* d_indices1_;
    float* d_conv2_out_;
    float* d_pool2_out_;
    float* d_indices2_;
    float* d_conv3_out_;
    float* d_up1_out_;
    float* d_conv4_out_;
    float* d_up2_out_;
    float* d_conv5_out_;
    
    // Device pointers for gradients
    float* d_grad_conv5_out_;
    float* d_grad_up2_out_;
    float* d_grad_conv4_out_;
    float* d_grad_up1_out_;
    float* d_grad_conv3_out_;
    float* d_grad_pool2_out_;
    float* d_grad_conv2_out_;
    float* d_grad_pool1_out_;
    float* d_grad_conv1_out_;
    
    float* d_grad_conv1_weights_;
    float* d_grad_conv1_bias_;
    float* d_grad_conv2_weights_;
    float* d_grad_conv2_bias_;
    float* d_grad_conv3_weights_;
    float* d_grad_conv3_bias_;
    float* d_grad_conv4_weights_;
    float* d_grad_conv4_bias_;
    float* d_grad_conv5_weights_;
    float* d_grad_conv5_bias_;
    
    float* d_loss_;  // For computing loss on device
    
    int current_batch_size_;
    
    void initialize_weights();
    void allocate_device_memory(int batch_size);
    void free_device_memory();
    
    void forward_gpu_optimized(int batch_size);
    void backward_gpu_optimized(int batch_size);
    void update_weights_gpu(float learning_rate, int batch_size);
    
    float compute_loss_gpu(int batch_size);
};

#endif // AUTOENCODER_GPU_OPTIMIZED_2_H
