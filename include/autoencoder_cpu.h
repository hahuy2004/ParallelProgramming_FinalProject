#ifndef AUTOENCODER_CPU_H
#define AUTOENCODER_CPU_H

#include <vector>
#include <string>

// Autoencoder CPU: Mô hình nén và giải nén ảnh
// Kiến trúc: Encoder (2 conv + pool) -> Latent (8x8x128) -> Decoder (2 conv + upsample)
// Dùng để trích xuất features từ ảnh CIFAR-10
class AutoencoderCPU {
public:
    AutoencoderCPU();
    ~AutoencoderCPU();

    // Training: Huấn luyện autoencoder để học nén và giải nén ảnh
    void train(const std::vector<float>& train_images, 
               int num_images,
               int batch_size, 
               int epochs, 
               float learning_rate);
    
    // Trích xuất features: Chỉ chạy encoder để lấy latent representation
    void extract_features(const std::vector<float>& images,
                         int num_images,
                         std::vector<float>& features);
    
    // Lưu và load weights của model
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);
    
    // Tính loss: MSE giữa ảnh gốc và ảnh reconstruct
    float compute_loss(const std::vector<float>& input,
                      const std::vector<float>& output,
                      int batch_size);

private:
    // Thông số kiến trúc mạng
    static constexpr int INPUT_H = 32;  // Chiều cao ảnh input
    static constexpr int INPUT_W = 32;  // Chiều rộng ảnh input
    static constexpr int INPUT_C = 3;  // Số channels (RGB)
    
    // Kích thước các layer
    static constexpr int CONV1_FILTERS = 256;  // Số filters ở conv layer 1
    static constexpr int CONV2_FILTERS = 128;  // Số filters ở conv layer 2
    static constexpr int LATENT_H = 8;  // Chiều cao latent space
    static constexpr int LATENT_W = 8;  // Chiều rộng latent space
    static constexpr int LATENT_C = 128;  // Số channels latent space
    static constexpr int LATENT_DIM = LATENT_H * LATENT_W * LATENT_C;  // Tổng dim: 8192
    
    // Weights và biases của các layer
    std::vector<float> conv1_weight;  // [256, 3, 3, 3] - Encoder layer 1
    std::vector<float> conv1_bias;     // [256]
    std::vector<float> conv2_weight;  // [128, 256, 3, 3] - Encoder layer 2
    std::vector<float> conv2_bias;     // [128]
    std::vector<float> conv3_weight;  // [128, 128, 3, 3] - Decoder layer 1
    std::vector<float> conv3_bias;     // [128]
    std::vector<float> conv4_weight;  // [256, 128, 3, 3] - Decoder layer 2
    std::vector<float> conv4_bias;     // [256]
    std::vector<float> conv5_weight;  // [3, 256, 3, 3] - Output layer
    std::vector<float> conv5_bias;     // [3]
    
    // Buffers để lưu activation của mỗi layer (cho 1 batch)
    std::vector<float> conv1_out;
    std::vector<float> pool1_out;
    std::vector<float> conv2_out;
    std::vector<float> pool2_out;  // Latent representation
    std::vector<float> conv3_out;
    std::vector<float> up1_out;
    std::vector<float> conv4_out;
    std::vector<float> up2_out;
    std::vector<float> conv5_out;  // Reconstruction
    
    // Gradient buffers
    std::vector<float> grad_conv5;
    std::vector<float> grad_up2;
    std::vector<float> grad_conv4;
    std::vector<float> grad_up1;
    std::vector<float> grad_conv3;
    std::vector<float> grad_pool2;
    std::vector<float> grad_conv2;
    std::vector<float> grad_pool1;
    std::vector<float> grad_conv1;
    
    std::vector<float> conv1_weight_grad;
    std::vector<float> conv1_bias_grad;
    std::vector<float> conv2_weight_grad;
    std::vector<float> conv2_bias_grad;
    std::vector<float> conv3_weight_grad;
    std::vector<float> conv3_bias_grad;
    std::vector<float> conv4_weight_grad;
    std::vector<float> conv4_bias_grad;
    std::vector<float> conv5_weight_grad;
    std::vector<float> conv5_bias_grad;
    
    int current_batch_size;
    
    // Các operations cơ bản
    void initialize_weights();  // Khởi tạo weights (Xavier initialization)
    void allocate_buffers(int batch_size);  // Cấp phát bộ nhớ cho buffers
    
    // Forward pass: Tính toán output từ input
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
    
    // Backward pass: Tính gradient để update weights
    void conv2d_backward(const float* grad_output, float* grad_input,
                        float* grad_weights, float* grad_bias,
                        const float* input, const float* weights,
                        int batch, int in_h, int in_w, int in_c,
                        int out_c, int kernel_size, int stride, int padding);
    
    void relu_backward(const float* grad_output, float* grad_input,
                      const float* output, int size);
    
    void maxpool2d_backward(const float* grad_output, float* grad_input,
                           const float* input, const float* output,
                           int batch, int h, int w, int c,
                           int pool_size, int stride);
    
    void upsample2d_backward(const float* grad_output, float* grad_input,
                            int batch, int in_h, int in_w, int c,
                            int scale_factor);
    
    // Hàm chính cho training
    void forward(const float* input, int batch_size);  // Forward pass toàn bộ mạng
    void backward(const float* input, int batch_size);  // Backward pass toàn bộ mạng
    void update_weights(float learning_rate);  // Update weights bằng gradient descent
};

#endif // AUTOENCODER_CPU_H
