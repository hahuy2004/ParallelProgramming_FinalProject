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

bool Cifar10Loader::load_batch(const std::string& filepath,
                                std::vector<float>& images,
                                std::vector<uint8_t>& labels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Each record: 1 byte label + 3072 bytes image (32x32x3)
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
        
        // First byte is label
        labels.push_back(buffer[0]);
        
        // Next 3072 bytes are image data (R, G, B channels)
        // CIFAR-10 format: all R values, then all G values, then all B values
        size_t start_idx = images.size();
        images.resize(start_idx + IMAGE_PIXELS);
        
        // Convert to HWC format and normalize to [0, 1]
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            for (int h = 0; h < IMAGE_SIZE; ++h) {
                for (int w = 0; w < IMAGE_SIZE; ++w) {
                    int cifar_idx = 1 + c * IMAGE_SIZE * IMAGE_SIZE + h * IMAGE_SIZE + w;
                    int hwc_idx = start_idx + h * IMAGE_SIZE * NUM_CHANNELS + w * NUM_CHANNELS + c;
                    images[hwc_idx] = buffer[cifar_idx] / 255.0f;
                }
            }
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
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(train_indices_.begin(), train_indices_.end(), gen);
}
