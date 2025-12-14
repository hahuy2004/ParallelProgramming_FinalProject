#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 2: Naive GPU Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ==================== CONFIGURATION ==================== //
    // GPU PHASE: Full training với 50000 ảnh, 20 epochs
    int batch_size = 64;  
    int epochs = 5;  // Change to 20 for full training
    float learning_rate = 0.001f;
    int num_train_images = 50000; // Change to 50000 for full training
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    //=====================================================//



    // ===================== Load Dataset ==================== //
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
    
    //=====================================================//



    //==================== TRAINING ====================//
    std::cout << "\n========== Training GPU Autoencoder (Naive) =========" << std::endl;

    AutoencoderGPU autoencoder;
    
    auto train_start = std::chrono::high_resolution_clock::now();

    autoencoder.train(train_images,
                     num_train_images,
                     batch_size,
                     epochs,
                     learning_rate);
    
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    // Save weights
    std::cout << "\n=== Saving Model ===" << std::endl;
    std::string weights_path = "weights/autoencoder_gpu_naive.weights";
    autoencoder.save_weights(weights_path);
    
    std::cout << "\n=== Phase 2 Completed ===" << std::endl;
    std::cout << "Total training time: " << train_time << " seconds" << std::endl;
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    return 0;
}

