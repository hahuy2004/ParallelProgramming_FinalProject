#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu_optimized_1.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 3.1: Optimized GPU version 1" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    // GPU OPTIMIZED PHASE: Full training với 50000 ảnh, 20 epochs
    int batch_size = 64;  
    int epochs = 7;  // Change to 20 for full training
    float learning_rate = 0.001f;
    // int num_train_images = 1000; // Change to 50000 for full training
    
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
    std::cout << "Expected time: ~" << (loader.get_train_size() * epochs / 1024 * 77.5 / 60) 
              << " minutes (estimated from test runs)" << std::endl;
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

    //--------------------------Thống kê Version 1-------------------------------------------------
    std::cout << "\n=== Training Version 1 Summary ===" << std::endl;
    std::cout << "Total training time version 1: " << train_time_1 << " seconds" << std::endl;
    std::cout << "Time per epoch version 1: " << (train_time_1 / epochs) << " seconds" << std::endl;

    // Save weights
    std::string weights_path_1 = "weights/autoencoder_gpu_optimized_1.weights";
    autoencoder_ver_1.save_weights(weights_path_1);
        
    std::cout << "\n=== Phase 3.1 Completed ===" << std::endl;
    std::cout << "Weights version 1 saved to: " << weights_path_1 << std::endl;
    std::cout << "Memory Optimization Training Time: " << train_time_1 << "s" << std::endl;

    return 0;
}
