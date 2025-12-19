#include "../include/cifar10_loader.h"
#include "../include/autoencoder_cpu.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 1: CPU Baseline Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ===== CONFIGURATION =====
    // Đường dẫn tới CIFAR-10 dataset (có thể truyền qua argument)
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    // Hyperparameters cho training
    int batch_size = 32;  // Số ảnh trong mỗi batch
    int epochs = 1;  // Số epoch (chỉ 1 epoch cho sanity check & benchmarking)
    
    float learning_rate = 0.001f;
    
    // ===== LOAD DATASET =====
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
    
    // ===== TRAINING CONFIGURATION =====
    // Số lượng ảnh để train (có thể thay đổi)
    // int num_train_images = loader.get_train_size();  // 50,000 ảnh (toàn bộ dataset)
    int num_train_images = 128;  // Train 500 ảnh
    
    // Khởi tạo và train autoencoder
    std::cout << "\n=== Training CPU Autoencoder (Sanity Check) ===" << std::endl;
    std::cout << "NOTE: CPU Phase - Training with " << num_train_images << " images, " 
              << epochs << " epoch for sanity check & benchmarking" << std::endl;
    AutoencoderCPU autoencoder;
    
    // Đo thời gian training
    auto train_start = std::chrono::high_resolution_clock::now();
    
    // Bắt đầu training
    autoencoder.train(train_images, 
                     num_train_images,  // Số lượng ảnh đã chọn ở trên
                     batch_size,
                     epochs,
                     learning_rate);
    
    // Kết thúc training và tính thời gian
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    std::cout << "\n=== Training Summary (CPU Baseline - Sanity Check) ===" << std::endl;
    std::cout << "Total training time: " << train_time << " seconds" << std::endl;
    std::cout << "Time per epoch: " << (train_time / epochs) << " seconds" << std::endl;
    std::cout << "Images processed: " << num_train_images << std::endl;
    
    // ===== SAVE MODEL =====
    // Lưu weights đã train để dùng sau (extract features hoặc continue training)
    std::string weights_path = "weights/autoencoder_cpu.bin";
    autoencoder.save_weights(weights_path);
    
    std::cout << "\n=== Phase 1 Completed ===" << std::endl;
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    return 0;
}
