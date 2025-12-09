#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu_optimized_1.h"
#include "../include/autoencoder_gpu_optimized_2.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 3: Optimized GPU Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    // GPU OPTIMIZED PHASE: Full training với 50000 ảnh, 20 epochs
    int batch_size = 128;  // Larger batch size
    int epochs = 20;  // Full 20 epochs 
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
    
    // Create and train optimized GPU autoencoder
    std::cout << "\n=== Training GPU Autoencoder (optimized) ===" << std::endl;
    std::cout << "Full training with " << loader.get_train_size() 
              << " images, " << epochs << " epochs" << std::endl;
    
    //---------------------Memory Optimization---------------------
    AutoencoderGPUOptimized1 autoencoder_ver_1;
    
    auto train_start_1 = std::chrono::high_resolution_clock::now();
    
    autoencoder_ver_1.train(train_images,
                     loader.get_train_size(),
                     batch_size,
                     epochs,
                     learning_rate);
    
    auto train_end_1 = std::chrono::high_resolution_clock::now();
    float train_time_1 = std::chrono::duration<float>(train_end_1 - train_start_1).count();
    
    //---------------------Kernel-Level Optimization---------------------
    AutoencoderGPUOptimized2 autoencoder_ver_2;
    
    auto train_start_2 = std::chrono::high_resolution_clock::now();
    
    autoencoder_ver_2.train(train_images,
                     loader.get_train_size(),
                     batch_size,
                     epochs,
                     learning_rate);
    
    auto train_end_2 = std::chrono::high_resolution_clock::now();
    float train_time_2 = std::chrono::duration<float>(train_end_2 - train_start_2).count();


    //--------------------------Thống kê Version 1-------------------------------------------------
    std::cout << "\n=== Training Version 1 Summary ===" << std::endl;
    std::cout << "Total training time version 1: " << train_time_1 << " seconds" << std::endl;
    std::cout << "Time per epoch version 1: " << (train_time_1 / epochs) << " seconds" << std::endl;

    // Save weights
    std::string weights_path_1 = "weights/autoencoder_gpu_optimized_1.weights";
    autoencoder_ver_1.save_weights(weights_path_1);
    
    // Extract features
    std::cout << "\n=== Extracting Features Version 1 ===" << std::endl;
    std::vector<float> train_features_1;
    std::vector<float> test_features_1;
    
    auto extract_start_1 = std::chrono::high_resolution_clock::now();
    
    autoencoder_ver_1.extract_features(train_images, loader.get_train_size(), train_features_1);
    autoencoder_ver_1.extract_features(loader.get_test_images(), loader.get_test_size(), test_features_1);
    
    auto extract_end_1 = std::chrono::high_resolution_clock::now();
    float extract_time_1 = std::chrono::duration<float>(extract_end_1 - extract_start_1).count();
    
    std::cout << "Feature extraction completed" << std::endl;
    std::cout << "Training features: (" << loader.get_train_size() << ", 8192)" << std::endl;
    std::cout << "Test features: (" << loader.get_test_size() << ", 8192)" << std::endl;
    std::cout << "Time: " << extract_time_1 << " seconds" << std::endl;
    
    //--------------------------Thống kê Version 2-------------------------------------------------
    std::cout << "\n=== Training Version 2 Summary ===" << std::endl;
    std::cout << "Total training time version 2: " << train_time_2 << " seconds" << std::endl;
    std::cout << "Time per epoch version 2: " << (train_time_2 / epochs) << " seconds" << std::endl;

    // Save weights
    std::string weights_path_2 = "weights/autoencoder_gpu_optimized_2.weights";
    autoencoder_ver_2.save_weights(weights_path_2);
    
    // Extract features
    std::cout << "\n=== Extracting Features Version 2 ===" << std::endl;
    std::vector<float> train_features_2;
    std::vector<float> test_features_2;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    autoencoder_ver_2.extract_features(train_images, loader.get_train_size(), train_features_2);
    autoencoder_ver_2.extract_features(loader.get_test_images(), loader.get_test_size(), test_features_2);
    
    auto extract_end_2 = std::chrono::high_resolution_clock::now();
    float extract_time_2 = std::chrono::duration<float>(extract_end_2 - extract_start_2).count();
    
    std::cout << "Feature extraction completed" << std::endl;
    std::cout << "Training features: (" << loader.get_train_size() << ", 8192)" << std::endl;
    std::cout << "Test features: (" << loader.get_test_size() << ", 8192)" << std::endl;
    std::cout << "Time: " << extract_time_2 << " seconds" << std::endl;
    
    
    
    std::cout << "\n=== Phase 3 Completed ===" << std::endl;
    std::cout << "Weights version 1 saved to: " << weights_path_1 << std::endl;
    std::cout << "Weights version 2 saved to: " << weights_path_2 << std::endl;
    std::cout << "\nMemory Optimization Results:" << std::endl;
    std::cout << "  Training: " << train_time_1 << "s (Target: <600s)" << std::endl;
    std::cout << "  Feature extraction: " << extract_time_1 << "s (Target: <20s)" << std::endl;
    std::cout << "\nKernel Fusion Optimization Results:" << std::endl;
    std::cout << "  Training: " << train_time_2 << "s (Target: <600s)" << std::endl;
    std::cout << "  Feature extraction: " << extract_time_2 << "s (Target: <20s)" << std::endl;
    
    return 0;
}
