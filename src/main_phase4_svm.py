#!/usr/bin/env python3
"""
PHASE 4: SVM Classification using cuML (OPTIMIZED VERSION)
Optimized for Tesla T4 GPU:
- Fast I/O using np.fromfile
- Batch inference to prevent VRAM OOM
- Aggressive RAM cleanup
- Vectorized Confusion Matrix calculation
"""

import os
import sys
import numpy as np
import time
import struct
import gc  # Garbage Collector để dọn RAM

# Kiểm tra và import thư viện GPU
try:
    from cuml.svm import SVC
    # Thử import confusion_matrix từ cuml, nếu không có thì dùng sklearn (để tương thích ngược)
    try:
        from cuml.metrics import confusion_matrix
        USE_CUML_CM = True
    except ImportError:
        from sklearn.metrics import confusion_matrix
        USE_CUML_CM = False

    import cupy as cp
    CUML_AVAILABLE = True
    print(f"cuML Available. Using GPU-accelerated Confusion Matrix: {USE_CUML_CM}")
except ImportError:
    print("Warning: cuML not available. Please install cuML.")
    sys.exit(1)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_features(filepath):
    """
    Tối ưu hóa: Đọc trực tiếp binary block vào NumPy array (nhanh gấp 100 lần struct).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    with open(filepath, 'rb') as f:
        # Đọc header (2 số integer = 8 bytes)
        header = f.read(8)
        num_images = struct.unpack('i', header[:4])[0]
        feature_dim = struct.unpack('i', header[4:8])[0]

        # Đọc toàn bộ mảng float trong 1 lần đọc
        # count = số lượng phần tử float cần đọc
        features = np.fromfile(f, dtype=np.float32, count=num_images * feature_dim)
        features = features.reshape(num_images, feature_dim)

    print(f"Loaded {filepath}:")
    print(f"  - Samples: {num_images}")
    print(f"  - Feature dimension: {feature_dim}")

    return features, num_images, feature_dim

def load_labels(data_dir, is_train=True):
    """
    Load label CIFAR-10, dùng seek để nhảy qua dữ liệu ảnh (nhanh hơn read).
    """
    labels = []
    if is_train:
        # Load 5 training batches
        for i in range(1, 6):
            batch_file = os.path.join(data_dir, f"data_batch_{i}.bin")
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Training batch not found: {batch_file}")

            with open(batch_file, 'rb') as f:
                for _ in range(10000):
                    labels.append(struct.unpack('B', f.read(1))[0])
                    f.seek(3072, 1) # Nhảy qua 3072 bytes ảnh
    else:
        # Load test batch
        test_file = os.path.join(data_dir, "test_batch.bin")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test batch not found: {test_file}")

        with open(test_file, 'rb') as f:
            for _ in range(10000):
                labels.append(struct.unpack('B', f.read(1))[0])
                f.seek(3072, 1)

    return np.array(labels, dtype=np.int32)

def print_confusion_matrix(cm, class_names):
    """In Confusion Matrix đẹp."""
    print("\n=== Confusion Matrix ===")
    print(f"{'':>12}", end='')
    for name in class_names:
        print(f"{name[:10]:>12}", end='')
    print()

    for i, name in enumerate(class_names):
        print(f"{name[:10]:>12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end='')
        print()

def print_per_class_accuracy(cm, class_names):
    """In độ chính xác từng lớp."""
    print("\n=== Per-Class Accuracy ===")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total > 0:
            acc = 100.0 * cm[i, i] / total
            print(f"{name:>15}: {acc:>6.2f}%")
        else:
            print(f"{name:>15}: N/A")

def predict_in_batches(model, X_gpu, batch_size=4096):
    """
    Dự đoán theo lô (batch) để tránh tràn VRAM GPU và RAM CPU.
    """
    num_samples = X_gpu.shape[0]
    predictions = []

    # Dự đoán từng phần
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)

        # Lấy slice trực tiếp trên GPU (không copy về CPU)
        batch_X = X_gpu[i:end_idx]

        # Predict trả về array (có thể là cupy hoặc numpy tùy version cuml)
        batch_pred = model.predict(batch_X)

        predictions.append(batch_pred)

        # Clean up
        del batch_X

    # Nối các mảng lại (dùng thư viện phù hợp dựa trên kiểu dữ liệu trả về)
    if len(predictions) > 0:
        if isinstance(predictions[0], cp.ndarray):
            return cp.concatenate(predictions)
        else:
            return np.concatenate(predictions)
    return np.array([])

def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels,
                           C=1.0, kernel='rbf', gamma='scale', save_path=None):

    print("\n=== Step 1: Preparing Data for GPU ===")
    start_transfer = time.time()

    # 1. Chuyển Training Data lên GPU
    print("Moving Training Data to GPU...")
    X_train_gpu = cp.asarray(train_features, dtype=cp.float32)
    y_train_gpu = cp.asarray(train_labels, dtype=cp.int32)

    # [QUAN TRỌNG] Xóa ngay dữ liệu gốc trên RAM CPU để giải phóng bộ nhớ
    del train_features, train_labels
    gc.collect()

    # 2. Chuyển Test Data lên GPU
    print("Moving Test Data to GPU...")
    X_test_gpu = cp.asarray(test_features, dtype=cp.float32)
    y_test_gpu = cp.asarray(test_labels, dtype=cp.int32)

    # [QUAN TRỌNG] Xóa tiếp
    del test_features, test_labels
    gc.collect()

    transfer_time = time.time() - start_transfer
    print(f"Data transfer & Cleanup time: {transfer_time:.4f} seconds")
    print(f"Training set shape (GPU): {X_train_gpu.shape}")

    # Initialize SVM
    print(f"\n=== Step 2: Training SVM ===")
    print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma}")

    # cache_size=2000 (2GB) để tận dụng RAM GPU, giúp train nhanh hơn
    svm = SVC(C=C, kernel=kernel, gamma=gamma, cache_size=2000, max_iter=100, tol=1e-3, verbose=True)

    train_start = time.time()
    svm.fit(X_train_gpu, y_train_gpu)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.4f} seconds")

    # Evaluate on training set
    print("\n=== Step 3: Evaluating on Training Set (Batched) ===")
    train_pred_start = time.time()

    # Dùng hàm batch để không bị OOM
    train_predictions = predict_in_batches(svm, X_train_gpu, batch_size=4096)

    # Nếu kết quả là numpy array (do version cũ), chuyển sang cupy để tính toán nhanh
    if not isinstance(train_predictions, cp.ndarray):
        train_predictions = cp.asarray(train_predictions)

    train_accuracy = cp.mean(train_predictions == y_train_gpu) * 100
    train_pred_time = time.time() - train_pred_start

    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Training prediction time: {train_pred_time:.4f} seconds")

    # Evaluate on test set
    print("\n=== Step 4: Evaluating on Test Set (Batched) ===")
    test_pred_start = time.time()

    test_predictions = predict_in_batches(svm, X_test_gpu, batch_size=4096)

    if not isinstance(test_predictions, cp.ndarray):
        test_predictions = cp.asarray(test_predictions)

    test_accuracy = cp.mean(test_predictions == y_test_gpu) * 100
    test_pred_time = time.time() - test_pred_start

    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Test prediction time: {test_pred_time:.4f} seconds")

    # Confusion Matrix Optimization
    print("\n=== Computing Confusion Matrix ===")
    if USE_CUML_CM:
        # Tính trực tiếp trên GPU (cực nhanh)
        cm_gpu = confusion_matrix(y_test_gpu, test_predictions)
        # Chuyển kết quả (chỉ 10x10) về CPU để in
        cm = cp.asnumpy(cm_gpu)
    else:
        # Fallback về sklearn (CPU) nếu cuML lỗi, nhưng cần chuyển dữ liệu về CPU trước
        y_test_cpu = cp.asnumpy(y_test_gpu)
        test_pred_cpu = cp.asnumpy(test_predictions)
        cm = confusion_matrix(y_test_cpu, test_pred_cpu)

    print_confusion_matrix(cm, CLASS_NAMES)
    print_per_class_accuracy(cm, CLASS_NAMES)

    # Save model if requested
    if save_path:
        print(f"\n=== Saving Model ===")
        # Note: cuML models can be saved using pickle or joblib
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(svm, f)
        print(f"Model saved to: {save_path}")

    # Summary
    print("\n" + "=" * 50)
    print("=== SUMMARY ===")
    print("=" * 50)
    print(f"Data transfer time:       {transfer_time:.4f}s")
    print(f"Training time:            {train_time:.4f}s")
    print(f"Training prediction time: {train_pred_time:.4f}s")
    print(f"Test prediction time:     {test_pred_time:.4f}s")
    print(f"Total time:               {transfer_time + train_time + train_pred_time + test_pred_time:.4f}s")
    print(f"\nTraining accuracy:        {train_accuracy:.2f}%")
    print(f"Test accuracy:            {test_accuracy:.2f}%")
    print("=" * 50)

    return svm, test_accuracy

def main():
    # Hardcoded parameters
    mode = 'gpu_naive'
    data_dir = 'cifar-10-batches-bin'
    # Thay thế bằng đường dẫn tới thư mục features đúng
    features_base_dir = '/content/ParallelProgramming_FinalProject/features'
    kernel = 'rbf'
    C = 1.0
    gamma = 'scale'
    save_model = None

    print("=" * 60)
    print("PHASE 4: SVM Classification (Optimized for T4/P100)")
    print("=" * 60)

    feature_dir = os.path.join(features_base_dir, mode)
    train_feature_path = os.path.join(feature_dir, 'train_features.bin')
    test_feature_path = os.path.join(feature_dir, 'test_features.bin')

    # Load Features
    print(f"\n=== Loading Features ===")
    try:
        train_features, num_train, train_dim = load_features(train_feature_path)
        test_features, num_test, test_dim = load_features(test_feature_path)

        # Kiểm tra sanity
        if train_dim != test_dim:
            raise ValueError(f"Mismatch dim: train={train_dim}, test={test_dim}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load Labels
    print(f"\n=== Loading Labels ===")
    try:
        train_labels = load_labels(data_dir, is_train=True)
        test_labels = load_labels(data_dir, is_train=False)

        # Cắt label cho khớp với số lượng feature (phòng trường hợp feature extract thiếu)
        train_labels = train_labels[:num_train]
        test_labels = test_labels[:num_test]

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Train
    try:
        # Xử lý gamma
        gamma_val = gamma
        if isinstance(gamma, str) and gamma not in ['scale', 'auto']:
            gamma_val = float(gamma)

        train_and_evaluate_svm(
            train_features, train_labels,
            test_features, test_labels,
            C=C, kernel=kernel, gamma=gamma_val,
            save_path=save_model
        )

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()