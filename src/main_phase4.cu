#include "../include/cifar10_loader.h"
#include "../include/autoencoder_cpu.h"
#include "../include/autoencoder_gpu.h"
#include "../include/autoencoder_gpu_optimized.h"
#include "../include/svm_classifier.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cctype>

void print_confusion_matrix(const std::vector<std::vector<int>>& cm,
                           const std::vector<std::string>& class_names) {
    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    std::cout << std::setw(12) << " ";
    for (const auto& name : class_names) {
        std::cout << std::setw(12) << name.substr(0, 10);
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < cm.size(); ++i) {
        std::cout << std::setw(12) << class_names[i].substr(0, 10);
        for (size_t j = 0; j < cm[i].size(); ++j) {
            std::cout << std::setw(12) << cm[i][j];
        }
        std::cout << std::endl;
    }
}

void print_per_class_accuracy(const std::vector<std::vector<int>>& cm,
                              const std::vector<std::string>& class_names) {
    std::cout << "\n=== Per-Class Accuracy ===" << std::endl;
    for (size_t i = 0; i < cm.size(); ++i) {
        int total = 0;
        for (int val : cm[i]) total += val;
        float acc = (total > 0) ? (100.0f * cm[i][i] / total) : 0.0f;
        std::cout << std::setw(15) << class_names[i] << ": " 
                  << std::fixed << std::setprecision(2) << acc << "%" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "PHASE 4: Complete Pipeline with SVM" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Configuration
    std::string data_dir = "cifar-10-batches-bin";
    std::string mode = "optimized";  // Default: optimized GPU
    
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
    if (mode != "cpu" && mode != "naive" && mode != "optimized") {
        std::cerr << "Invalid mode: " << mode << std::endl;
        std::cerr << "Usage: " << argv[0] << " [data_dir] [--mode cpu|naive|optimized]" << std::endl;
        std::cerr << "Default mode: optimized" << std::endl;
        return 1;
    }
    
    std::cout << "Mode: " << mode << std::endl;
    
    std::vector<std::string> class_names = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    
    // Load CIFAR-10 dataset
    std::cout << "\n=== Step 1: Loading CIFAR-10 Dataset ===" << std::endl;
    Cifar10Loader loader(data_dir);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    
    std::cout << "Training images: " << loader.get_train_size() << std::endl;
    std::cout << "Test images: " << loader.get_test_size() << std::endl;
    
    // Number of samples to use (CPU mode uses limited samples)
    int num_train_samples = loader.get_train_size();
    int num_test_samples = loader.get_test_size();
    
    if (mode == "cpu") {
        num_train_samples = 500;  // CPU was trained with only 500 images
        num_test_samples = 1000;  // Use 1000 test images for CPU mode
        std::cout << "\nNOTE: CPU mode - using " << num_train_samples 
                  << " training samples and " << num_test_samples << " test samples" << std::endl;
    }
    
    // Extract features
    std::cout << "\n=== Step 2: Loading Autoencoder and Extracting Features ===" << std::endl;
    std::vector<float> train_features;
    std::vector<float> test_features;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    if (mode == "cpu") {
        std::cout << "Using CPU Autoencoder" << std::endl;
        AutoencoderCPU autoencoder;
        std::string weights_path = "weights/autoencoder_cpu.weights";
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(loader.get_train_images(), 
                                    num_train_samples, 
                                    train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(loader.get_test_images(),
                                    num_test_samples,
                                    test_features);
    } else if (mode == "naive") {
        std::cout << "Using Naive GPU Autoencoder" << std::endl;
        AutoencoderGPU autoencoder;
        std::string weights_path = "weights/autoencoder_gpu_naive.weights";
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(loader.get_train_images(), 
                                    num_train_samples, 
                                    train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(loader.get_test_images(),
                                    num_test_samples,
                                    test_features);
    } else {  // optimized
        std::cout << "Using Optimized GPU Autoencoder" << std::endl;
        AutoencoderGPUOptimized autoencoder;
        std::string weights_path = "weights/autoencoder_gpu_optimized.weights";
        autoencoder.load_weights(weights_path);
        
        std::cout << "Extracting training features..." << std::endl;
        autoencoder.extract_features(loader.get_train_images(), 
                                    num_train_samples, 
                                    train_features);
        
        std::cout << "Extracting test features..." << std::endl;
        autoencoder.extract_features(loader.get_test_images(),
                                    num_test_samples,
                                    test_features);
    }
    
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();
    
    std::cout << "Feature extraction completed in " << extract_time << " seconds" << std::endl;
    std::cout << "Training features shape: (" << num_train_samples << ", 8192)" << std::endl;
    std::cout << "Test features shape: (" << num_test_samples << ", 8192)" << std::endl;
    
    // Train SVM
    std::cout << "\n=== Step 3: Training SVM Classifier ===" << std::endl;
    SVMClassifier svm;
    
    auto svm_train_start = std::chrono::high_resolution_clock::now();
    
    if (mode == "cpu") {
        // CPU mode: use subset of labels
        std::vector<uint8_t> train_labels_subset(
            loader.get_train_labels().begin(),
            loader.get_train_labels().begin() + num_train_samples
        );
        svm.train(train_features, train_labels_subset, num_train_samples, 8192);
    } else {
        // GPU modes: use full training labels
        svm.train(train_features, loader.get_train_labels(), num_train_samples, 8192);
    }
    
    auto svm_train_end = std::chrono::high_resolution_clock::now();
    float svm_train_time = std::chrono::duration<float>(svm_train_end - svm_train_start).count();
    
    std::cout << "SVM training completed in " << svm_train_time << " seconds" << std::endl;
    
    // Save SVM model
    std::string svm_model_path = (mode == "cpu") ? "weights/svm_model_cpu.model" : 
                                  (mode == "naive") ? "weights/svm_model_naive.model" :
                                  "weights/svm_model_optimized.model";
    svm.save_model(svm_model_path);
    
    // Predict on test set
    std::cout << "\n=== Step 4: Evaluating on Test Set ===" << std::endl;
    std::vector<uint8_t> predictions;
    
    auto predict_start = std::chrono::high_resolution_clock::now();
    
    svm.predict(test_features, num_test_samples, 8192, predictions);
    
    auto predict_end = std::chrono::high_resolution_clock::now();
    float predict_time = std::chrono::duration<float>(predict_end - predict_start).count();
    
    std::cout << "Prediction completed in " << predict_time << " seconds" << std::endl;
    
    // Evaluate
    float accuracy;
    std::vector<uint8_t> test_labels_for_eval;
    
    if (mode == "cpu") {
        // CPU mode: use subset of test labels
        test_labels_for_eval.assign(
            loader.get_test_labels().begin(),
            loader.get_test_labels().begin() + num_test_samples
        );
    } else {
        // GPU modes: use full test labels
        test_labels_for_eval = loader.get_test_labels();
    }
    
    accuracy = svm.evaluate(predictions, test_labels_for_eval);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Final Results ===" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
    
    // Compute and display confusion matrix
    std::vector<std::vector<int>> confusion_matrix;
    svm.compute_confusion_matrix(predictions, test_labels_for_eval,
                                 10, confusion_matrix);
    
    print_confusion_matrix(confusion_matrix, class_names);
    print_per_class_accuracy(confusion_matrix, class_names);
    
    // Performance summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Feature extraction time: " << extract_time << "s" << std::endl;
    std::cout << "SVM training time: " << svm_train_time << "s" << std::endl;
    std::cout << "SVM prediction time: " << predict_time << "s" << std::endl;
    std::cout << "Total pipeline time: " << (extract_time + svm_train_time + predict_time) << "s" << std::endl;
    
    return 0;
}
