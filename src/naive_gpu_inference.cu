#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing the reconstruction of autoencoder" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ===================== Load Dataset ==================== //
    // Load CIFAR-10 dataset
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }
    std::cout << "\n=== Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    
    std::cout << "\nDataset loaded successfully!" << std::endl;
    std::cout << "Training images: " << loader.get_train_size() << std::endl;
    std::cout << "Test images: " << loader.get_test_size() << std::endl;
    
    //=====================================================//



    // ===================== Initialize Autoencoder ==================== //
    AutoencoderGPU autoencoder;
    autoencoder.load_weights("weights/autoencoder_gpu_naive.weights");
    
    // Use test images for reconstruction
    const auto& test_images = loader.get_test_images();
    int start_idx = 100;           // Starting image index
    int num_samples = 10;        // Number of images to inference
    
    std::cout << "\n========== Running Inference ==========" << std::endl;
    std::cout << "Inferencing images [" << start_idx << ", " << (start_idx + num_samples) << ")" << std::endl;
    
    std::vector<float> reconstructions;
    // Get only the slice of images we want
    const float* test_slice = &test_images[start_idx * 32 * 32 * 3];
    // Create a vector from the slice
    std::vector<float> test_slice_vec(test_slice, test_slice + num_samples * 32 * 32 * 3);
    autoencoder.infer(test_slice_vec, num_samples, reconstructions);
    //==================================================================//

    


    // ===================== Save Results for Visualization ==================== //
    std::cout << "\n========== Saving Results ===========" << std::endl;
    
    system("mkdir -p results/naive_gpu_inference");

    std::ofstream orig_file("results/naive_gpu_inference/original_images.bin", std::ios::binary);
    std::ofstream recon_file("results/naive_gpu_inference/reconstructed_images.bin", std::ios::binary);
    
    if (!orig_file.is_open() || !recon_file.is_open()) {
        std::cerr << "Failed to open output files! Make sure results/ directory exists." << std::endl;
        return 1;
    }
    
    // Save original and reconstructed images (same slice that was inferred)
    orig_file.write(reinterpret_cast<const char*>(test_slice), 
                    num_samples * 32 * 32 * 3 * sizeof(float));
    recon_file.write(reinterpret_cast<const char*>(reconstructions.data()), 
                     num_samples * 32 * 32 * 3 * sizeof(float));
    
    orig_file.close();
    recon_file.close();
    
    std::cout << "Saved " << num_samples << " image pairs to results/naive_gpu_inference/" << std::endl;
    std::cout << "  - results/naive_gpu_inference/original_images.bin" << std::endl;
    std::cout << "  - results/naive_gpu_inference/reconstructed_images.bin" << std::endl;
     //==================================================================//



    return 0;
}

