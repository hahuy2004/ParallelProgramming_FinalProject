#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <vector>
#include <string>
#include <cstdint>

class SVMClassifier {
public:
    SVMClassifier();
    ~SVMClassifier();

    // Train SVM on extracted features
    void train(const std::vector<float>& features,
              const std::vector<uint8_t>& labels,
              int num_samples,
              int feature_dim);
    
    // Predict labels for test features
    void predict(const std::vector<float>& features,
                int num_samples,
                int feature_dim,
                std::vector<uint8_t>& predictions);
    
    // Evaluate accuracy
    float evaluate(const std::vector<uint8_t>& predictions,
                  const std::vector<uint8_t>& true_labels);
    
    // Compute confusion matrix
    void compute_confusion_matrix(const std::vector<uint8_t>& predictions,
                                 const std::vector<uint8_t>& true_labels,
                                 int num_classes,
                                 std::vector<std::vector<int>>& confusion_matrix);
    
    // Save/Load model
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);

private:
    void* svm_model_;  // Opaque pointer to LIBSVM model
    void* svm_params_; // Opaque pointer to LIBSVM parameters
};

#endif // SVM_CLASSIFIER_H
