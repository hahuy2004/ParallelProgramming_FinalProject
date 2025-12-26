#!/usr/bin/env python3
"""
PHASE 4: SVM Classification using cuML
Train and evaluate SVM classifier using extracted features from different modes
(cpu, gpu_naive, gpu_optimized_1, gpu_optimized_2)
"""

import os
import sys
import argparse
import numpy as np
import time
import struct
import gc  # For aggressive RAM cleanup

# cuML imports and confusion_matrix selection
try:
    from cuml.svm import SVC
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
    print("Install with: conda install -c rapidsai -c conda-forge cuml")
    CUML_AVAILABLE = False
    sys.exit(1)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_features(filepath):
    """
    Optimized: Read binary block directly into NumPy array (much faster than struct).
    Format: [num_images (int), feature_dim (int), features (floats)]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    with open(filepath, 'rb') as f:
        header = f.read(8)
        num_images = struct.unpack('i', header[:4])[0]
        feature_dim = struct.unpack('i', header[4:8])[0]
        features = np.fromfile(f, dtype=np.float32, count=num_images * feature_dim)
        features = features.reshape(num_images, feature_dim)

    print(f"Loaded {filepath}:")
    print(f"  - Samples: {num_images}")
    print(f"  - Feature dimension: {feature_dim}")

    return features, num_images, feature_dim


def load_labels(data_dir, is_train=True):
    """
    Load CIFAR-10 labels using seek to skip image data (faster than read).
    """
    labels = []
    if is_train:
        for i in range(1, 6):
            batch_file = os.path.join(data_dir, f"data_batch_{i}.bin")
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Training batch not found: {batch_file}")
            with open(batch_file, 'rb') as f:
                for _ in range(10000):
                    labels.append(struct.unpack('B', f.read(1))[0])
                    f.seek(3072, 1)  # Skip image data
    else:
        test_file = os.path.join(data_dir, "test_batch.bin")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test batch not found: {test_file}")
        with open(test_file, 'rb') as f:
            for _ in range(10000):
                labels.append(struct.unpack('B', f.read(1))[0])
                f.seek(3072, 1)
    return np.array(labels, dtype=np.int32)


def print_confusion_matrix(cm, class_names):
    """Print confusion matrix in a formatted way."""
    print("\n=== Confusion Matrix ===")
    
    # Header
    print(f"{'':>12}", end='')
    for name in class_names:
        print(f"{name[:10]:>12}", end='')
    print()
    
    # Rows
    for i, name in enumerate(class_names):
        print(f"{name[:10]:>12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end='')
        print()


def print_per_class_accuracy(cm, class_names):
    """Print per-class accuracy."""
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
    Predict in batches to avoid VRAM OOM and reduce RAM usage.
    """
    num_samples = X_gpu.shape[0]
    predictions = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_X = X_gpu[i:end_idx]
        batch_pred = model.predict(batch_X)
        predictions.append(batch_pred)
        del batch_X
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

    # Move training data to GPU
    print("Moving Training Data to GPU...")
    X_train_gpu = cp.asarray(train_features, dtype=cp.float32)
    y_train_gpu = cp.asarray(train_labels, dtype=cp.int32)
    del train_features, train_labels
    gc.collect()

    # Move test data to GPU
    print("Moving Test Data to GPU...")
    X_test_gpu = cp.asarray(test_features, dtype=cp.float32)
    y_test_gpu = cp.asarray(test_labels, dtype=cp.int32)
    del test_features, test_labels
    gc.collect()

    transfer_time = time.time() - start_transfer
    print(f"Data transfer & Cleanup time: {transfer_time:.4f} seconds")
    print(f"Training set shape (GPU): {X_train_gpu.shape}")

    # Initialize SVM
    print(f"\n=== Step 2: Training SVM ===")
    print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma}")
    svm = SVC(C=C, kernel=kernel, gamma=gamma, cache_size=2000, max_iter=100, tol=1e-3, verbose=True)

    train_start = time.time()
    svm.fit(X_train_gpu, y_train_gpu)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.4f} seconds")

    # Evaluate on training set (batched)
    print("\n=== Step 3: Evaluating on Training Set (Batched) ===")
    train_pred_start = time.time()
    train_predictions = predict_in_batches(svm, X_train_gpu, batch_size=4096)
    if not isinstance(train_predictions, cp.ndarray):
        train_predictions = cp.asarray(train_predictions)
    train_accuracy = cp.mean(train_predictions == y_train_gpu) * 100
    train_pred_time = time.time() - train_pred_start
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Training prediction time: {train_pred_time:.4f} seconds")

    # Evaluate on test set (batched)
    print("\n=== Step 4: Evaluating on Test Set (Batched) ===")
    test_pred_start = time.time()
    test_predictions = predict_in_batches(svm, X_test_gpu, batch_size=4096)
    if not isinstance(test_predictions, cp.ndarray):
        test_predictions = cp.asarray(test_predictions)
    test_accuracy = cp.mean(test_predictions == y_test_gpu) * 100
    test_pred_time = time.time() - test_pred_start
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Test prediction time: {test_pred_time:.4f} seconds")

    # Confusion Matrix
    print("\n=== Computing Confusion Matrix ===")
    if USE_CUML_CM:
        cm_gpu = confusion_matrix(y_test_gpu, test_predictions)
        cm = cp.asnumpy(cm_gpu)
    else:
        y_test_cpu = cp.asnumpy(y_test_gpu)
        test_pred_cpu = cp.asnumpy(test_predictions)
        cm = confusion_matrix(y_test_cpu, test_pred_cpu)
    print_confusion_matrix(cm, CLASS_NAMES)
    print_per_class_accuracy(cm, CLASS_NAMES)

    # Save model if requested
    if save_path:
        print(f"\n=== Saving Model ===")
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

    return svm, float(test_accuracy)


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate SVM using cuML with extracted features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use gpu_optimized_1 features (default)
  python main_phase4_svm.py
  
  # Use cpu features
  python main_phase4_svm.py --mode cpu
  
  # Use gpu_optimized_2 features with custom SVM parameters
  python main_phase4_svm.py --mode gpu_optimized_2 --kernel rbf --C 10.0
  
  # Use different data directory
  python main_phase4_svm.py --data-dir /path/to/cifar-10-batches-bin
  
Available modes:
  - cpu: Use features extracted by CPU autoencoder
  - gpu_naive: Use features extracted by naive GPU autoencoder
  - gpu_optimized_1: Use features extracted by optimized GPU autoencoder v1
  - gpu_optimized_2: Use features extracted by optimized GPU autoencoder v2
        """
    )
    
    parser.add_argument('--mode', '-m', type=str, default='gpu_optimized_1',
                       choices=['cpu', 'gpu_naive', 'gpu_optimized_1', 'gpu_optimized_2'],
                       help='Mode for feature extraction (default: gpu_optimized_1)')
    
    parser.add_argument('--data-dir', type=str, default='cifar-10-batches-bin',
                       help='Path to CIFAR-10 dataset directory (default: cifar-10-batches-bin)')
    
    parser.add_argument('--features-dir', type=str, default='features',
                       help='Path to features directory (default: features)')
    
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'poly', 'rbf', 'sigmoid'],
                       help='SVM kernel type (default: rbf)')
    
    parser.add_argument('--C', type=float, default=1.0,
                       help='SVM regularization parameter (default: 1.0)')
    
    parser.add_argument('--gamma', type=str, default='scale',
                       help='Kernel coefficient (default: scale)')
    
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained model (optional)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 4: SVM Classification using cuML")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Features directory: {args.features_dir}")
    print(f"SVM Kernel: {args.kernel}")
    print(f"SVM C: {args.C}")
    print(f"SVM Gamma: {args.gamma}")
    
    # Construct feature paths based on mode
    if args.mode == 'cpu':
        feature_dir = os.path.join(args.features_dir, 'cpu')
    elif args.mode == 'gpu_naive':
        feature_dir = os.path.join(args.features_dir, 'gpu_naive')
    elif args.mode == 'gpu_optimized_1':
        feature_dir = os.path.join(args.features_dir, 'gpu_optimized_1')
    elif args.mode == 'gpu_optimized_2':
        feature_dir = os.path.join(args.features_dir, 'gpu_optimized_2')
    
    train_feature_path = os.path.join(feature_dir, 'train_features.bin')
    test_feature_path = os.path.join(feature_dir, 'test_features.bin')
    
    # Load features
    print(f"\n=== Loading Features ===")
    print(f"Feature directory: {feature_dir}")
    
    try:
        train_features, num_train, train_dim = load_features(train_feature_path)
        test_features, num_test, test_dim = load_features(test_feature_path)
        
        if train_dim != test_dim:
            raise ValueError(f"Feature dimension mismatch: train={train_dim}, test={test_dim}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nPlease extract features first using:")
        print(f"  ./build/phase4_extract_features --mode {args.mode}")
        sys.exit(1)
    
    # Load labels
    print(f"\n=== Loading Labels ===")
    print(f"Data directory: {args.data_dir}")
    
    try:
        train_labels = load_labels(args.data_dir, is_train=True)
        test_labels = load_labels(args.data_dir, is_train=False)
        
        # Ensure labels match features
        train_labels = train_labels[:num_train]
        test_labels = test_labels[:num_test]
        
        print(f"Loaded train labels: {len(train_labels)}")
        print(f"Loaded test labels: {len(test_labels)}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please ensure CIFAR-10 dataset is in: {args.data_dir}")
        sys.exit(1)
    
    # Train and evaluate SVM
    try:
        gamma = args.gamma
        if gamma not in ['scale', 'auto']:
            gamma = float(gamma)
        
        svm_model, accuracy = train_and_evaluate_svm(
            train_features, train_labels,
            test_features, test_labels,
            C=args.C,
            kernel=args.kernel,
            gamma=gamma,
            save_path=args.save_model
        )
        
        print(f"\n✓ SVM training and evaluation completed successfully!")
        print(f"✓ Final test accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"\nError during SVM training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if not CUML_AVAILABLE:
        print("cuML is required for this script.")
        print("Please install it using:")
        print("  conda install -c rapidsai -c conda-forge cuml")
        sys.exit(1)
    
    main()
