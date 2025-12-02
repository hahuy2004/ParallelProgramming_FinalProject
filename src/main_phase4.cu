#include "cifar10_loader.h"
#include "autoencoder_gpu.h"
#include "svm_classifier.h"
#include <iostream>
#include <chrono>
#include <iomanip>

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
    if (argc > 1) {
        data_dir = argv[1];
    }
    
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
    
    // Load or train autoencoder
    std::cout << "\n=== Step 2: Loading Autoencoder ===" << std::endl;
    AutoencoderGPU autoencoder;
    
    std::string weights_path = "weights/autoencoder_gpu_naive.weights";
    autoencoder.load_weights(weights_path);
    
    // Extract features
    std::cout << "\n=== Step 3: Extracting Features ===" << std::endl;
    std::vector<float> train_features;
    std::vector<float> test_features;
    
    auto extract_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Extracting training features..." << std::endl;
    autoencoder.extract_features(loader.get_train_images(), 
                                loader.get_train_size(), 
                                train_features);
    
    std::cout << "Extracting test features..." << std::endl;
    autoencoder.extract_features(loader.get_test_images(),
                                loader.get_test_size(),
                                test_features);
    
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();
    
    std::cout << "Feature extraction completed in " << extract_time << " seconds" << std::endl;
    std::cout << "Training features shape: (" << loader.get_train_size() << ", 8192)" << std::endl;
    std::cout << "Test features shape: (" << loader.get_test_size() << ", 8192)" << std::endl;
    
    // Train SVM
    std::cout << "\n=== Step 4: Training SVM Classifier ===" << std::endl;
    SVMClassifier svm;
    
    auto svm_train_start = std::chrono::high_resolution_clock::now();
    
    svm.train(train_features, loader.get_train_labels(),
             loader.get_train_size(), 8192);
    
    auto svm_train_end = std::chrono::high_resolution_clock::now();
    float svm_train_time = std::chrono::duration<float>(svm_train_end - svm_train_start).count();
    
    std::cout << "SVM training completed in " << svm_train_time << " seconds" << std::endl;
    
    // Save SVM model
    svm.save_model("weights/svm_model.model");
    
    // Predict on test set
    std::cout << "\n=== Step 5: Evaluating on Test Set ===" << std::endl;
    std::vector<uint8_t> predictions;
    
    auto predict_start = std::chrono::high_resolution_clock::now();
    
    svm.predict(test_features, loader.get_test_size(), 8192, predictions);
    
    auto predict_end = std::chrono::high_resolution_clock::now();
    float predict_time = std::chrono::duration<float>(predict_end - predict_start).count();
    
    std::cout << "Prediction completed in " << predict_time << " seconds" << std::endl;
    
    // Evaluate
    float accuracy = svm.evaluate(predictions, loader.get_test_labels());
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Final Results ===" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
    
    // Compute and display confusion matrix
    std::vector<std::vector<int>> confusion_matrix;
    svm.compute_confusion_matrix(predictions, loader.get_test_labels(),
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
