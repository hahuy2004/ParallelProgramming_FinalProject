#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <string>
#include <vector>

// Layer structure to hold weights and gradients
struct ConvLayer {
    float *d_weights;
    float *d_bias;
    float *d_grad_weights;
    float *d_grad_bias;
    int in_c, out_c, kernel_size;
    int weight_size, bias_size;
    
    ConvLayer(int in_channels, int out_channels, int k_size);
    ~ConvLayer();
};

// Autoencoder class
class Autoencoder {
private:
    // Encoder layers
    ConvLayer *conv1;  // 3 -> 256
    ConvLayer *conv2;  // 256 -> 128
    
    // Decoder layers
    ConvLayer *conv3;  // 128 -> 128
    ConvLayer *conv4;  // 128 -> 256
    ConvLayer *conv5;  // 256 -> 3
    
    // Intermediate activations and gradients
    float *d_enc1, *d_enc1_relu, *d_enc1_pool;
    float *d_enc2, *d_enc2_relu, *d_latent;
    float *d_dec1, *d_dec1_relu, *d_dec1_up;
    float *d_dec2, *d_dec2_relu, *d_dec2_up;
    float *d_output;
    
    // Gradients for backward pass
    float *d_grad_output;
    float *d_grad_dec2_up, *d_grad_dec2_relu, *d_grad_dec2;
    float *d_grad_dec1_up, *d_grad_dec1_relu, *d_grad_dec1;
    float *d_grad_latent;
    float *d_grad_enc2_relu, *d_grad_enc2, *d_grad_enc1_pool;
    float *d_grad_enc1_relu, *d_grad_enc1;
    
    // MaxPool indices
    float *d_pool1_indices, *d_pool2_indices;
    
    // Loss
    float *d_loss;
    
    int batch_size;
    float learning_rate;
    
public:
    Autoencoder(int batch, float lr = 0.001f);
    ~Autoencoder();
    
    // Forward pass
    float forward(const float* d_input, const float* d_target);
    
    // Backward pass
    void backward(const float* d_input, const float* d_target);
    
    // Update weights using SGD
    void update_weights();
    
    // Save weights to file
    void save_weights(const std::string& filename);
    
    // Load weights from file
    void load_weights(const std::string& filename);
    
    // Get output for inference
    void get_output(float* h_output);

    void train(const std::vector<float>& train_data, 
                      int num_samples, int epochs)
};

#endif // AUTOENCODER_H