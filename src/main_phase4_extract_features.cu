#include "../include/cifar10_loader.h"
#include "../include/autoencoder_cpu.h"
#include "../include/autoencoder_gpu.h"
#include "../include/autoencoder_gpu_optimized_1.h"
#include "../include/autoencoder_gpu_optimized_2.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cctype>
#include <sys/stat.h>
#include <sys/types.h>

void save_features(const std::string& filepath, 
                   const std::vector<float>& features,
                   int num_images,
                   int feature_dim) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to create file: " << filepath << std::endl;
        return;
    }
    
    // Write metadata
    out.write(reinterpret_cast<const char*>(&num_images), sizeof(int));
    out.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));
    
    // Write features
    out.write(reinterpret_cast<const char*>(features.data()), 
              features.size() * sizeof(float));
    
    out.close();
    std::cout << "Saved: " << filepath 
              << " (" << num_images << " samples, " 
              << feature_dim << " dims)" << std::endl;
}

void extract_and_save_features(const std::string& mode,
                               const std::string& weights_path,
                               const std::vector<float>& train_images,
                               const std::vector<float>& test_images,
                               int num_train_samples,
                               int num_test_samples,
                               const std::string& output_dir) {
    std::cout << "\n=== Extracting Features: " << mode << " ===" << std::endl;
    std::cout << "Weights: " << weights_path << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    
    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());
    
    std::vector<float> train_features;
    std::vector<float> test_features;
    const int FEATURE_DIM = 8192;  // Latent dimension
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    if (mode == "cpu") {
        AutoencoderCPU autoencoder;
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(train_images, num_train_samples, train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(test_images, num_test_samples, test_features);
        
    } else if (mode == "gpu_naive") {
        AutoencoderGPU autoencoder;
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(train_images, num_train_samples, train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(test_images, num_test_samples, test_features);
        
    } else if (mode == "gpu_optimized_1") {
        AutoencoderGPUOptimized1 autoencoder;
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(train_images, num_train_samples, train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(test_images, num_test_samples, test_features);
        
    } else if (mode == "gpu_optimized_2") {
        AutoencoderGPUOptimized2 autoencoder;
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(train_images, num_train_samples, train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(test_images, num_test_samples, test_features);
    }
    
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();
    
    std::cout << "Feature extraction completed in " << extract_time << " seconds" << std::endl;
    
    // Save features
    std::string train_path = output_dir + "/train_features.bin";
    std::string test_path = output_dir + "/test_features.bin";
    
    save_features(train_path, train_features, num_train_samples, FEATURE_DIM);
    save_features(test_path, test_features, num_test_samples, FEATURE_DIM);
    
    std::cout << "Extraction time: " << extract_time << "s" << std::endl;
    std::cout << "---" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 4: Extract Features (Single Model)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration - default mode is gpu_optimized_1
    std::string mode = "gpu_optimized_1";
    std::string data_dir = "cifar-10-batches-bin";
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" || arg == "-m") {
            if (i + 1 < argc) {
                mode = argv[++i];
                // Convert to lowercase
                std::transform(mode.begin(), mode.end(), mode.begin(),
                             [](unsigned char c){ return std::tolower(c); });
            }
        } else {
            data_dir = arg;
        }
    }

    // Validate mode
    if (mode != "cpu" && mode != "gpu_naive" && mode != "gpu_optimized_1" && mode != "gpu_optimized_2") {
        std::cerr << "Invalid mode: " << mode << std::endl;
        std::cerr << "Usage: " << argv[0] << " [--mode cpu|gpu_naive|gpu_optimized_1|gpu_optimized_2]" << std::endl;
        std::cerr << "Default mode: gpu_optimized_1" << std::endl;
        return 1;
    }
    
    std::cout << "Mode: " << mode << std::endl;
    
    // Load CIFAR-10 dataset
    std::cout << "\n=== Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    
    std::cout << "Training images: " << loader.get_train_size() << std::endl;
    std::cout << "Test images: " << loader.get_test_size() << std::endl;
    
    const auto& train_images = loader.get_train_images();
    const auto& test_images = loader.get_test_images();
    
    // Determine number of samples and weights path based on mode
    int num_train_samples = loader.get_train_size();
    int num_test_samples = loader.get_test_size();
    std::string weights_path;
    
    if (mode == "cpu") {
        num_train_samples = 500;  // CPU was trained with only 500 images
        num_test_samples = 1000;  // CPU: only extract 1000 test images
        std::cout << "\nNOTE: CPU mode - using " << num_train_samples 
                  << " training samples and " << num_test_samples << " test samples" << std::endl;
        weights_path = "weights/autoencoder_cpu.weights";
    } else if (mode == "gpu_naive") {
        weights_path = "weights/autoencoder_gpu_naive.weights";
    } else if (mode == "gpu_optimized_1") {
        weights_path = "weights/autoencoder_gpu_optimized_1.weights";
    } else if (mode == "gpu_optimized_2") {
        weights_path = "weights/autoencoder_gpu_optimized_2.weights";
    }
    
    // Check if weights file exists
    if (!std::ifstream(weights_path).good()) {
        std::cerr << "Weights file not found: " << weights_path << std::endl;
        return 1;
    }
    
    // Create output directory based on mode
    std::string output_dir = "features/" + mode;
    
    // Extract and save features using the dedicated function
    std::cout << "\n=== Extraction Summary ===" << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Training samples: " << num_train_samples << std::endl;
    std::cout << "Test samples: " << num_test_samples << std::endl;
    std::cout << "Feature dimension: 8192" << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    
    extract_and_save_features(mode, weights_path, train_images, test_images,
                              num_train_samples, num_test_samples, output_dir);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Feature Extraction Completed ===" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
