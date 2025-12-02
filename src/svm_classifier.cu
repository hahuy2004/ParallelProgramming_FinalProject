#include "svm_classifier.h"
#include "../third_party/libsvm/svm.h"
#include <iostream>
#include <algorithm>
#include <cstdint>

SVMClassifier::SVMClassifier() {
    svm_model_ = nullptr;
    
    // Initialize SVM parameters
    svm_parameter* params = new svm_parameter();
    params->svm_type = C_SVC;
    params->kernel_type = RBF;
    params->degree = 3;
    params->gamma = 0;  // Will be set to 1/num_features
    params->coef0 = 0;
    params->nu = 0.5;
    params->cache_size = 100;
    params->C = 10.0;
    params->eps = 1e-3;
    params->p = 0.1;
    params->shrinking = 1;
    params->probability = 0;
    params->nr_weight = 0;
    params->weight_label = nullptr;
    params->weight = nullptr;
    
    svm_params_ = params;
}

SVMClassifier::~SVMClassifier() {
    if (svm_model_) {
        svm_free_and_destroy_model(reinterpret_cast<svm_model**>(&svm_model_));
    }
    if (svm_params_) {
        delete reinterpret_cast<svm_parameter*>(svm_params_);
    }
}

void SVMClassifier::train(const std::vector<float>& features,
                          const std::vector<uint8_t>& labels,
                          int num_samples,
                          int feature_dim) {
    std::cout << "Training SVM classifier..." << std::endl;
    std::cout << "Samples: " << num_samples << ", Features: " << feature_dim << std::endl;
    
    svm_parameter* params = reinterpret_cast<svm_parameter*>(svm_params_);
    
    // Set gamma if not specified
    if (params->gamma == 0) {
        params->gamma = 1.0 / feature_dim;
    }
    
    // Prepare LIBSVM problem
    svm_problem prob;
    prob.l = num_samples;
    prob.y = new double[num_samples];
    prob.x = new svm_node*[num_samples];
    
    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = labels[i];
        
        // Convert dense features to sparse format
        prob.x[i] = new svm_node[feature_dim + 1];
        for (int j = 0; j < feature_dim; ++j) {
            prob.x[i][j].index = j + 1;  // LIBSVM uses 1-based indexing
            prob.x[i][j].value = features[i * feature_dim + j];
        }
        prob.x[i][feature_dim].index = -1;  // End marker
    }
    
    // Check parameters
    const char* error_msg = svm_check_parameter(&prob, params);
    if (error_msg) {
        std::cerr << "SVM parameter error: " << error_msg << std::endl;
        // Free memory
        for (int i = 0; i < num_samples; ++i) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
        delete[] prob.y;
        return;
    }
    
    // Train model
    std::cout << "Training SVM (this may take a while)..." << std::endl;
    svm_model* model = svm_train(&prob, params);
    
    if (svm_model_) {
        svm_free_and_destroy_model(reinterpret_cast<svm_model**>(&svm_model_));
    }
    svm_model_ = model;
    
    // Free memory
    for (int i = 0; i < num_samples; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;
    
    std::cout << "SVM training completed" << std::endl;
}

void SVMClassifier::predict(const std::vector<float>& features,
                            int num_samples,
                            int feature_dim,
                            std::vector<uint8_t>& predictions) {
    if (!svm_model_) {
        std::cerr << "SVM model not trained!" << std::endl;
        return;
    }
    
    predictions.resize(num_samples);
    svm_model* model = reinterpret_cast<svm_model*>(svm_model_);
    
    svm_node* x = new svm_node[feature_dim + 1];
    
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            x[j].index = j + 1;
            x[j].value = features[i * feature_dim + j];
        }
        x[feature_dim].index = -1;
        
        double pred = svm_predict(model, x);
        predictions[i] = static_cast<uint8_t>(pred);
    }
    
    delete[] x;
}

float SVMClassifier::evaluate(const std::vector<uint8_t>& predictions,
                              const std::vector<uint8_t>& true_labels) {
    if (predictions.size() != true_labels.size()) {
        std::cerr << "Size mismatch in evaluation!" << std::endl;
        return 0.0f;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == true_labels[i]) {
            correct++;
        }
    }
    
    float accuracy = static_cast<float>(correct) / predictions.size();
    return accuracy;
}

void SVMClassifier::compute_confusion_matrix(const std::vector<uint8_t>& predictions,
                                             const std::vector<uint8_t>& true_labels,
                                             int num_classes,
                                             std::vector<std::vector<int>>& confusion_matrix) {
    confusion_matrix.assign(num_classes, std::vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int true_label = true_labels[i];
        int pred_label = predictions[i];
        if (true_label < num_classes && pred_label < num_classes) {
            confusion_matrix[true_label][pred_label]++;
        }
    }
}

void SVMClassifier::save_model(const std::string& filepath) {
    if (!svm_model_) {
        std::cerr << "No model to save!" << std::endl;
        return;
    }
    
    svm_model* model = reinterpret_cast<svm_model*>(svm_model_);
    if (svm_save_model(filepath.c_str(), model) == 0) {
        std::cout << "SVM model saved to " << filepath << std::endl;
    } else {
        std::cerr << "Failed to save SVM model" << std::endl;
    }
}

void SVMClassifier::load_model(const std::string& filepath) {
    if (svm_model_) {
        svm_free_and_destroy_model(reinterpret_cast<svm_model**>(&svm_model_));
    }
    
    svm_model* model = svm_load_model(filepath.c_str());
    if (model) {
        svm_model_ = model;
        std::cout << "SVM model loaded from " << filepath << std::endl;
    } else {
        std::cerr << "Failed to load SVM model from " << filepath << std::endl;
    }
}
