#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

class Cifar10Loader {
public:
    Cifar10Loader(const std::string& data_dir);
    ~Cifar10Loader();

    // Load all training and test data
    bool load();
    
    // Get data
    const std::vector<float>& get_train_images() const { return train_images_; }
    const std::vector<uint8_t>& get_train_labels() const { return train_labels_; }
    const std::vector<float>& get_test_images() const { return test_images_; }
    const std::vector<uint8_t>& get_test_labels() const { return test_labels_; }
    
    // Get batch
    void get_batch(int batch_idx, int batch_size, std::vector<float>& batch_images);
    void shuffle_training_data();
    
    // Dataset info
    size_t get_train_size() const { return train_labels_.size(); }
    size_t get_test_size() const { return test_labels_.size(); }
    
    static constexpr int IMAGE_SIZE = 32;
    static constexpr int NUM_CHANNELS = 3;
    static constexpr int IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;
    static constexpr int NUM_CLASSES = 10;

private:
    bool load_batch(const std::string& filepath, 
                   std::vector<float>& images, 
                   std::vector<uint8_t>& labels);
    
    std::string data_dir_;
    std::vector<float> train_images_;      // Normalized [0,1]
    std::vector<uint8_t> train_labels_;
    std::vector<float> test_images_;       // Normalized [0,1]
    std::vector<uint8_t> test_labels_;
    std::vector<int> train_indices_;       // For shuffling
};

#endif // CIFAR10_LOADER_H
