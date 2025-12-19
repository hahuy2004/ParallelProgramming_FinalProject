#include "../include/cifar10_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

Cifar10Loader::Cifar10Loader(const std::string& data_dir) 
    : data_dir_(data_dir) {
}

Cifar10Loader::~Cifar10Loader() {
}

bool Cifar10Loader::load() {
    std::cout << "Loading CIFAR-10 dataset from " << data_dir_ << std::endl;
    
    // Load training batches (5 files)
    for (int i = 1; i <= 5; ++i) {
        std::string filepath = data_dir_ + "/data_batch_" + std::to_string(i) + ".bin";
        if (!load_batch(filepath, train_images_, train_labels_)) {
            std::cerr << "Failed to load " << filepath << std::endl;
            return false;
        }
        std::cout << "Loaded training batch " << i << std::endl;
    }
    
    // Load test batch
    std::string test_filepath = data_dir_ + "/test_batch.bin";
    if (!load_batch(test_filepath, test_images_, test_labels_)) {
        std::cerr << "Failed to load " << test_filepath << std::endl;
        return false;
    }
    std::cout << "Loaded test batch" << std::endl;
    
    // Initialize indices for shuffling
    train_indices_.resize(train_labels_.size());
    for (size_t i = 0; i < train_indices_.size(); ++i) {
        train_indices_[i] = i;
    }
    
    std::cout << "Training images: " << train_labels_.size() << std::endl;
    std::cout << "Test images: " << test_labels_.size() << std::endl;
    
    return true;
}

// Đọc 1 file batch từ CIFAR-10
// Format: mỗi record = 1 byte label + 3072 bytes ảnh (32x32x3)
bool Cifar10Loader::load_batch(const std::string& filepath,
                                std::vector<float>& images,
                                std::vector<uint8_t>& labels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Mỗi record: 1 byte label + 3072 bytes ảnh (32x32x3)
    const int record_size = 1 + IMAGE_PIXELS;
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    int num_records = file_size / record_size;
    
    std::vector<uint8_t> buffer(record_size);
    
    for (int i = 0; i < num_records; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), record_size);
        
        if (!file) {
            std::cerr << "Error reading record " << i << std::endl;
            return false;
        }
        
        // Byte đầu tiên là label (0-9)
        labels.push_back(buffer[0]);
        
        // 3072 bytes tiếp theo là dữ liệu ảnh (R, G, B channels)
        // Format CIFAR-10: tất cả giá trị R, rồi tất cả G, rồi tất cả B
        // Giữ nguyên format CHW (Channel-Height-Width) vì GPU kernels expect CHW
        size_t start_idx = images.size();
        images.resize(start_idx + IMAGE_PIXELS);
        
        // Normalize về [0, 1] và giữ format CHW
        for (int i = 0; i < IMAGE_PIXELS; ++i) {
            images[start_idx + i] = buffer[1 + i] / 255.0f;
        }
    }
    
    file.close();
    return true;
}

void Cifar10Loader::get_batch(int batch_idx, int batch_size, 
                               std::vector<float>& batch_images) {
    int start_idx = batch_idx * batch_size;
    int end_idx = std::min(start_idx + batch_size, (int)train_labels_.size());
    int actual_batch_size = end_idx - start_idx;
    
    batch_images.resize(actual_batch_size * IMAGE_PIXELS);
    
    for (int i = 0; i < actual_batch_size; ++i) {
        int img_idx = train_indices_[start_idx + i];
        std::memcpy(&batch_images[i * IMAGE_PIXELS],
                   &train_images_[img_idx * IMAGE_PIXELS],
                   IMAGE_PIXELS * sizeof(float));
    }
}

void Cifar10Loader::shuffle_training_data() {
    // static std::random_device rd;
    // static std::mt19937 gen(rd()); 
    static std::mt19937 gen(42);       // Fixed seed - reproducible results
    std::shuffle(train_indices_.begin(), train_indices_.end(), gen);
}
