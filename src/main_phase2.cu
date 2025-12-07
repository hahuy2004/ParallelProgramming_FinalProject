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
    int batch_size = 64;  // Larger batch size for GPU
    int epochs = 20;  // Full 20 epochs theo yêu cầu
    float learning_rate = 0.001f;
    
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
    std::cout << "NOTE: GPU Phase - Full training with " << loader.get_train_size() 
              << " images, " << epochs << " epochs" << std::endl;
    AutoencoderGPU autoencoder;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    autoencoder.train(train_images,
                     loader.get_train_size(),
                     batch_size,
                     epochs,
                     learning_rate);
    
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "Total training time: " << train_time << " seconds" << std::endl;
    std::cout << "Time per epoch: " << (train_time / epochs) << " seconds" << std::endl;
    
    // Save weights
    std::string weights_path = "weights/autoencoder_gpu_naive.weights";
    autoencoder.save_weights(weights_path);
    
    // Extract features
    std::cout << "\n=== Extracting Features ===" << std::endl;
    std::vector<float> train_features;
    std::vector<float> test_features;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    autoencoder.extract_features(train_images, loader.get_train_size(), train_features);
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
