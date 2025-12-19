#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <sys/stat.h> 
#include <cstring>

#include "../include/cifar10_loader.h"
#include "../include/autoencoder_gpu.h"
#include "../include/autoencoder_gpu_optimized_1.h"
#include "../include/autoencoder_gpu_optimized_2.h"
#include "../include/autoencoder_cpu.h"


// Lưu ảnh định dạng PPM từ dữ liệu CHW
// Ảnh sẽ được visualize bằng code python sau
void save_ppm_chw(const float* data, int width, int height, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    ofs << "P6\n" << width << " " << height << "\n255\n";
    
    int plane_size = width * height;
    for (int i = 0; i < plane_size; ++i) {
        float r = data[0 * plane_size + i] * 255.0f;
        float g = data[1 * plane_size + i] * 255.0f;
        float b = data[2 * plane_size + i] * 255.0f;

        ofs.put(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r))));
        ofs.put(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g))));
        ofs.put(static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b))));
    }
    ofs.close();
}


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model_type> <num_images> [data_dir]\n", argv[0]);
        return 1;
    }

    std::string model_type = argv[1];
    int num_to_visualize = std::atoi(argv[2]);
    std::string data_dir = (argc > 3) ? argv[3] : "cifar-10-batches-bin";

    // Kiểm tra model_type hợp lệ
    if (model_type != "cpu" && model_type != "gpu_naive" && 
        model_type != "gpu_opt1" && model_type != "gpu_opt2") {
        std::cerr << "Error: Invalid model type '" << model_type << "'" << std::endl;
        std::cout << "Valid options: cpu, gpu_naive, gpu_opt1, gpu_opt2" << std::endl;
        return 1;
    }

    // Kiểm tra số lượng ảnh
    if (num_to_visualize <= 0) {
        std::cerr << "Error: Number of images must be positive" << std::endl;
        return 1;
    }

    std::string output_dir = "results/reconstruction_" + model_type;
    std::string weights_path;

    if (model_type == "cpu") {
        weights_path = "weights/autoencoder_cpu.bin";
    } else if (model_type == "gpu_naive") {
        weights_path = "weights/autoencoder_gpu_naive.bin";
    } else if (model_type == "gpu_opt1") {
        weights_path = "weights/autoencoder_gpu_optimized_1.bin";
    } else if (model_type == "gpu_opt2") {
        weights_path = "weights/autoencoder_gpu_optimized_2.bin";
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "CIFAR-10 Autoencoder Reconstruction" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Init data loader 
    std::cout << "\nLoading CIFAR-10 dataset..." << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load CIFAR-10 data!" << std::endl;
        return 1;
    }
    std::cout << "Dataset loaded: " << loader.get_test_size() << " test images" << std::endl;
    std::cout << "\nLoading weights from: " << weights_path << std::endl;
    
    // Lấy ảnh test 
    const std::vector<float>& test_images = loader.get_test_images();
    const int IMAGE_PIXELS = 3 * 32 * 32; // 3072

    // Tạo thư mục lưu kết quả nếu chưa có
    system(("mkdir -p " + output_dir).c_str());


    // Bắt đầu vòng lặp reconstruct và lưu ảnh
    std::cout << "Starting reconstruction loop..." << std::endl;

    if (model_type == "cpu") {
        AutoencoderCPU ae;
        ae.load_weights(weights_path);

        for (int i = 0; i < num_to_visualize; ++i) {
            const float* input_ptr = &test_images[i * IMAGE_PIXELS];
            std::vector<float> reconstructed(IMAGE_PIXELS);

            ae.reconstruct(input_ptr, reconstructed.data());

            std::string orig_path = output_dir + "/img_" + std::to_string(i) + "_orig.ppm";
            std::string recon_path = output_dir + "/img_" + std::to_string(i) + "_recon.ppm";

            save_ppm_chw(input_ptr, 32, 32, orig_path);
            save_ppm_chw(reconstructed.data(), 32, 32, recon_path);
            
            std::cout << "  Finished image " << (i+1) << "/" << num_to_visualize << std::endl;
        }

    } else if (model_type == "gpu_naive") {
        AutoencoderGPU ae;
        ae.load_weights(weights_path);

        for (int i = 0; i < num_to_visualize; ++i) {
            const float* input_ptr = &test_images[i * IMAGE_PIXELS];
            std::vector<float> reconstructed(IMAGE_PIXELS);

            ae.reconstruct(input_ptr, reconstructed.data());

            std::string orig_path = output_dir + "/img_" + std::to_string(i) + "_orig.ppm";
            std::string recon_path = output_dir + "/img_" + std::to_string(i) + "_recon.ppm";

            save_ppm_chw(input_ptr, 32, 32, orig_path);
            save_ppm_chw(reconstructed.data(), 32, 32, recon_path);
            
            std::cout << "  Finished image " << (i+1) << "/" << num_to_visualize << std::endl;
        }

    } else if (model_type == "gpu_opt1") {
        AutoencoderGPUOptimized1 ae;
        ae.load_weights(weights_path);

        for (int i = 0; i < num_to_visualize; ++i) {
            const float* input_ptr = &test_images[i * IMAGE_PIXELS];
            std::vector<float> reconstructed(IMAGE_PIXELS);

            ae.reconstruct(input_ptr, reconstructed.data());

            std::string orig_path = output_dir + "/img_" + std::to_string(i) + "_orig.ppm";
            std::string recon_path = output_dir + "/img_" + std::to_string(i) + "_recon.ppm";

            save_ppm_chw(input_ptr, 32, 32, orig_path);
            save_ppm_chw(reconstructed.data(), 32, 32, recon_path);
            
            std::cout << "  Finished image " << (i+1) << "/" << num_to_visualize << std::endl;
        }

    } else if (model_type == "gpu_opt2") {
        AutoencoderGPUOptimized2 ae;
        ae.load_weights(weights_path);

        for (int i = 0; i < num_to_visualize; ++i) {
            const float* input_ptr = &test_images[i * IMAGE_PIXELS];
            std::vector<float> reconstructed(IMAGE_PIXELS);

            ae.reconstruct(input_ptr, reconstructed.data());

            std::string orig_path = output_dir + "/img_" + std::to_string(i) + "_orig.ppm";
            std::string recon_path = output_dir + "/img_" + std::to_string(i) + "_recon.ppm";

            save_ppm_chw(input_ptr, 32, 32, orig_path);
            save_ppm_chw(reconstructed.data(), 32, 32, recon_path);
            
            std::cout << "  Finished image " << (i+1) << "/" << num_to_visualize << std::endl;
        }
    }

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Success! Images are saved in: " << output_dir << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}