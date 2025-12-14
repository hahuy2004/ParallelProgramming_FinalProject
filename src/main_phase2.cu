#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 2: Naive GPU Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    // GPU PHASE: Full training với 50000 ảnh, 20 epochs
    int batch_size = 32;  
    int epochs = 4;  // Change to 20 for full training
    float learning_rate = 0.001f;
    int num_train_images = 128; // Change to 50000 for full training
    
    // Load CIFAR-10 dataset
    std::cout << "\n=== Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    
    const auto& train_images = loader.get_train_images();
    
    std::cout << "\nDataset loaded successfully!" << std::endl;
    std::cout << "Training images: " << loader.get_train_size() << std::endl;
    std::cout << "Test images: " << loader.get_test_size() << std::endl;
    
    // Create and train GPU autoencoder
    std::cout << "\n=== Training GPU Autoencoder (Naive) ===" << std::endl;
    std::cout << "NOTE: GPU Phase - Training with " << loader.get_train_size() 
              << " images over " << epochs << " epochs" << std::endl;
    std::cout << "Expected time: ~" << (loader.get_train_size() * epochs / 1024 * 77.5 / 60) 
              << " minutes (estimated from test runs)" << std::endl;
    AutoencoderGPU autoencoder(batch_size, learning_rate);
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    autoencoder.train(train_images, num_train_images, epochs);
    
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    // Save weights
    std::cout << "=== Saving Model ===" << std::endl;
    std::string weights_path = "weights/autoencoder_gpu_naive.weights";
    autoencoder.save_weights(weights_path);
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    // Extract features
    std::cout << "\n=== Extracting Features ===" << std::endl;
    std::cout << "Extracting 8192-dimensional features from encoder..." << std::endl;
    std::vector<float> train_features;
    std::vector<float> test_features;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "  - Extracting training features..." << std::endl;
    autoencoder.extract_features(train_images, loader.get_train_size(), train_features);
    std::cout << "  - Extracting test features..." << std::endl;
    autoencoder.extract_features(loader.get_test_images(), loader.get_test_size(), test_features);
    
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();
    
    std::cout << "Feature extraction completed" << std::endl;
    std::cout << "Training features: (" << loader.get_train_size() << ", 8192)" << std::endl;
    std::cout << "Test features: (" << loader.get_test_size() << ", 8192)" << std::endl;
    std::cout << "Time: " << extract_time << " seconds" << std::endl;
    
    std::cout << "\n=== Phase 2 Completed ===" << std::endl;
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    return 0;
}
