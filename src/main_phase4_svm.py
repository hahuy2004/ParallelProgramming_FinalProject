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
from pathlib import Path

# cuML imports
try:
    from cuml.svm import SVC
    import cupy as cp
    CUML_AVAILABLE = True
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
    Load features from binary file.
    Format: [num_images (int), feature_dim (int), features (floats)]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read metadata
        num_images = struct.unpack('i', f.read(4))[0]
        feature_dim = struct.unpack('i', f.read(4))[0]
        
        # Read features
        num_values = num_images * feature_dim
        features = np.array(struct.unpack(f'{num_values}f', f.read(num_values * 4)))
        features = features.reshape(num_images, feature_dim)
    
    print(f"Loaded {filepath}:")
    print(f"  - Samples: {num_images}")
    print(f"  - Feature dimension: {feature_dim}")
    
    return features, num_images, feature_dim


def load_labels(data_dir, is_train=True):
    """
    Load CIFAR-10 labels from binary files.
    """
    labels = []
    
    if is_train:
        # Load all 5 training batches
        for i in range(1, 6):
            batch_file = os.path.join(data_dir, f"data_batch_{i}.bin")
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Training batch not found: {batch_file}")
            
            with open(batch_file, 'rb') as f:
                for _ in range(10000):  # Each batch has 10000 images
                    label = struct.unpack('B', f.read(1))[0]
                    f.read(3072)  # Skip image data (32x32x3)
                    labels.append(label)
    else:
        # Load test batch
        test_file = os.path.join(data_dir, "test_batch.bin")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test batch not found: {test_file}")
        
        with open(test_file, 'rb') as f:
            for _ in range(10000):  # Test batch has 10000 images
                label = struct.unpack('B', f.read(1))[0]
                f.read(3072)  # Skip image data
                labels.append(label)
    
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


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels, 
                          C=1.0, kernel='rbf', gamma='scale', save_path=None):
    """
    Train SVM using cuML and evaluate on test set.
    
    Args:
        train_features: Training features (numpy array)
        train_labels: Training labels (numpy array)
        test_features: Test features (numpy array)
        test_labels: Test labels (numpy array)
        C: Regularization parameter
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        gamma: Kernel coefficient ('scale' or 'auto' or float)
        save_path: Path to save model (optional)
    """
    print("\n=== Step 1: Preparing Data for GPU ===")
    
    # Convert to CuPy arrays for GPU processing
    print("Transferring data to GPU...")
    start_transfer = time.time()
    
    X_train_gpu = cp.asarray(train_features, dtype=cp.float32)
    y_train_gpu = cp.asarray(train_labels, dtype=cp.int32)
    X_test_gpu = cp.asarray(test_features, dtype=cp.float32)
    y_test_gpu = cp.asarray(test_labels, dtype=cp.int32)
    
    transfer_time = time.time() - start_transfer
    print(f"Data transfer time: {transfer_time:.4f} seconds")
    
    print(f"Training set shape: {X_train_gpu.shape}")
    print(f"Test set shape: {X_test_gpu.shape}")
    
    # Initialize SVM
    print(f"\n=== Step 2: Training SVM ===")
    print(f"Parameters:")
    print(f"  - Kernel: {kernel}")
    print(f"  - C: {C}")
    print(f"  - Gamma: {gamma}")
    
    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        cache_size=1024,  # Cache size in MB
        max_iter=100,
        tol=1e-3,
        verbose=True
    )
    
    # Train
    print("\nTraining SVM on GPU...")
    train_start = time.time()
    
    svm.fit(X_train_gpu, y_train_gpu)
    
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.4f} seconds")
    
    # Evaluate on training set
    print("\n=== Step 3: Evaluating on Training Set ===")
    train_pred_start = time.time()
    
    train_predictions = svm.predict(X_train_gpu)
    train_predictions_cpu = cp.asnumpy(train_predictions)
    
    train_pred_time = time.time() - train_pred_start
    
    train_accuracy = np.mean(train_predictions_cpu == train_labels) * 100
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Training prediction time: {train_pred_time:.4f} seconds")
    
    # Evaluate on test set
    print("\n=== Step 4: Evaluating on Test Set ===")
    test_pred_start = time.time()
    
    test_predictions = svm.predict(X_test_gpu)
    test_predictions_cpu = cp.asnumpy(test_predictions)
    
    test_pred_time = time.time() - test_pred_start
    
    test_accuracy = np.mean(test_predictions_cpu == test_labels) * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Test prediction time: {test_pred_time:.4f} seconds")
    
    # Confusion matrix
    cm = compute_confusion_matrix(test_labels, test_predictions_cpu)
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
