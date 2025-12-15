#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

// Class để load và quản lý dataset CIFAR-10
// Hỗ trợ đọc file binary, normalize ảnh, và shuffle data
class Cifar10Loader {
public:
    // Constructor: Khởi tạo loader với đường dẫn tới thư mục chứa data
    Cifar10Loader(const std::string& data_dir);
    ~Cifar10Loader();

    // Load toàn bộ training và test data từ file binary
    bool load();
    
    // Lấy data đã load (trả về reference để tránh copy)
    const std::vector<float>& get_train_images() const { return train_images_; }  // Ảnh train (đã normalize)
    const std::vector<uint8_t>& get_train_labels() const { return train_labels_; }  // Label train
    const std::vector<float>& get_test_images() const { return test_images_; }  // Ảnh test
    const std::vector<uint8_t>& get_test_labels() const { return test_labels_; }  // Label test
    
    // Lấy 1 batch ảnh để training
    void get_batch(int batch_idx, int batch_size, std::vector<float>& batch_images);
    // Shuffle training data (để tránh overfitting)
    void shuffle_training_data();
    
    // Lấy thông tin về dataset
    size_t get_train_size() const { return train_labels_.size(); }  // Số lượng ảnh train
    size_t get_test_size() const { return test_labels_.size(); }  // Số lượng ảnh test
    
    // Thông số cố định của CIFAR-10
    static constexpr int IMAGE_SIZE = 32;  // Kích thước ảnh: 32x32
    static constexpr int NUM_CHANNELS = 3;  // 3 kênh màu (RGB)
    static constexpr int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;  // Tổng số pixels: 3072
    static constexpr int NUM_CLASSES = 10;  // 10 loại đối tượng

private:
    // Đọc 1 file batch (helper function)
    bool load_batch(const std::string& filepath, 
                   std::vector<float>& images, 
                   std::vector<uint8_t>& labels);
    
    // Data members
    std::string data_dir_;  // Đường dẫn tới thư mục data
    std::vector<float> train_images_;      // Ảnh training (đã normalize [0,1])
    std::vector<uint8_t> train_labels_;  // Label training
    std::vector<float> test_images_;       // Ảnh test (đã normalize [0,1])
    std::vector<uint8_t> test_labels_;  // Label test
    std::vector<int> train_indices_;       // Index để shuffle data
};

#endif // CIFAR10_LOADER_H
