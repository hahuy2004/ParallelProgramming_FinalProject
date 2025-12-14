#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu.h"
#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 2: Naive GPU Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ==================== CONFIGURATION ==================== //
    std::string data_dir = "cifar-10-batches-bin";
    std::string weights_path = "weights/autoencoder_gpu_naive.weights";
    int epochs = 5;
    int batch_size = 64;
    float learning_rate = 0.001f;
    int max_train_images = 50000;  // Set to smaller number for debugging
    
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) weights_path = argv[2];
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) learning_rate = std::stof(argv[5]);
    if (argc > 6) max_train_images = std::stoi(argv[6]);
    
    std::cout << "Data dir: " << data_dir << std::endl;
    std::cout << "Weights:  " << weights_path << std::endl;
    std::cout << "Epochs:   " << epochs << std::endl;
    std::cout << "Batch:    " << batch_size << std::endl;
    std::cout << "LR:       " << learning_rate << std::endl;
    std::cout << "Max train:" << max_train_images << std::endl;
    //=====================================================//

    // ===================== Load Dataset ==================== //
    std::cout << "\n=== Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }

    const auto& train_images = loader.get_train_images();
    
    std::cout << "Loaded train images: " << loader.get_train_size() << std::endl;
    std::cout << "Loaded test images:  " << loader.get_test_size() << std::endl;
    
    int num_train_images = loader.get_train_size();
    if (max_train_images > 0 && max_train_images < num_train_images) {
        num_train_images = max_train_images;
        std::cout << "Using " << num_train_images << " images (debug limit)" << std::endl;
    }
    
    const int IMAGE_PIXELS = 3072;  // 32*32*3
    //=====================================================//

    //==================== TRAINING ====================//
    std::cout << "\n=== Training GPU Autoencoder (Naive) ===" << std::endl;

    AutoencoderGPU ae;
    
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        
        float epoch_loss = 0.0f;
        int seen = 0;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        for (int start = 0; start < num_train_images; start += batch_size) {
            int end = std::min(start + batch_size, num_train_images);
            float batch_loss = 0.0f;
            
            // Train on each image in batch
            for (int i = start; i < end; i++) {
                // train_images is a flat vector: [img0_pixels, img1_pixels, ...]
                // Each image has 3072 pixels
                const float* img_data = &train_images[i * IMAGE_PIXELS];
                batch_loss += ae.train_step(img_data, learning_rate);
            }
            
            batch_loss /= (end - start);
            epoch_loss += batch_loss;
            seen++;
            
            int batch_num = start / batch_size + 1;
            if (batch_num == 1 || batch_num % 10 == 0) {
                std::cout << "  Batch " << batch_num << " loss: " << batch_loss << std::endl;
            }
        }
        
        epoch_loss /= seen;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_s = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch avg loss: " << epoch_loss << " | time: " << epoch_s << "s" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    std::cout << "\nTotal training time: " << total_s << "s" << std::endl;
    
    // Save weights
    std::cout << "\n=== Saving Model ===" << std::endl;
    if (!ae.save_weights(weights_path)) {
        std::cerr << "Failed to save weights!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Phase 2 Completed ===" << std::endl;
    std::cout << "Weights saved to: " << weights_path << std::endl;
    
    return 0;
}

