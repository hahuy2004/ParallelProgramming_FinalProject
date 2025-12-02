#include "cifar10_loader.h"
#include "autoencoder_cpu.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 1: CPU Baseline Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    // CPU PHASE: Sanity check với toàn bộ dataset, 1 epoch
    int batch_size = 32;
    int epochs = 1;  // Chỉ 1 epoch cho sanity check & benchmarking
    
    // Số lượng ảnh train:
    // - Mặc định: Toàn bộ 50,000 ảnh (để sanity check đầy đủ)
    // - Test nhanh: Uncomment dòng dưới để chỉ dùng 300 ảnh
    // int num_train_images = 300;  // <-- Uncomment để test nhanh
    
    float learning_rate = 0.001f;
    
    // Load CIFAR-10 dataset
    std::cout << "\n=== Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    
    const auto& train_images = loader.get_train_images();
    const auto& train_labels = loader.get_train_labels();
    
    std::cout << "\nDataset loaded successfully!" << std::endl;
    std::cout << "Full training images: " << loader.get_train_size() << std::endl;
    std::cout << "Test images: " << loader.get_test_size() << std::endl;
    std::cout << "Image size: " << Cifar10Loader::IMAGE_SIZE << "x" 
              << Cifar10Loader::IMAGE_SIZE << "x" 
              << Cifar10Loader::NUM_CHANNELS << std::endl;
    
    // Số lượng ảnh train (mặc định: toàn bộ dataset)
    // int num_train_images = loader.get_train_size();  // 50,000 ảnh (comment để dùng custom)
    int num_train_images = 500;  // <-- Chỉ train với 5,000 ảnh
    
    // Create and train autoencoder
    std::cout << "\n=== Training CPU Autoencoder (Sanity Check) ===" << std::endl;
    std::cout << "NOTE: CPU Phase - Training with " << num_train_images << " images, " 
              << epochs << " epoch for sanity check & benchmarking" << std::endl;
    AutoencoderCPU autoencoder;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    autoencoder.train(train_images, 
                     num_train_images,  // Toàn bộ dataset hoặc 300 ảnh (tùy config)
                     batch_size,
                     epochs,  // Chỉ 1 epoch
                     learning_rate);
    
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    std::cout << "\n=== Training Summary (CPU Baseline - Sanity Check) ===" << std::endl;
    std::cout << "Total training time: " << train_time << " seconds" << std::endl;
    std::cout << "Time per epoch: " << (train_time / epochs) << " seconds" << std::endl;
    std::cout << "Images processed: " << num_train_images << std::endl;
    
    // Save weights
    std::string weights_path = "weights/autoencoder_cpu.weights";
    autoencoder.save_weights(weights_path);
    
    // Extract features for a small subset (for demonstration)
    std::cout << "\n=== Extracting Features ===" << std::endl;
    int num_test_samples = 1000;  // Extract features for first 1000 samples
    std::vector<float> features;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    autoencoder.extract_features(train_images, num_test_samples, features);
    
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();
    
    std::cout << "Feature extraction completed" << std::endl;
    std::cout << "Extracted features shape: (" << num_test_samples << ", 8192)" << std::endl;
    std::cout << "Time: " << extract_time << " seconds" << std::endl;
    
    std::cout << "\n=== Phase 1 Completed ===" << std::endl;
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    return 0;
}
