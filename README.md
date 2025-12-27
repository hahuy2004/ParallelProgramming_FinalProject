# CIFAR-10 Autoencoder + SVM với CUDA

---

## 1. Tổng quan

Đây là đồ án cuối kỳ môn Lập trình song song (CSC14120) - Khoa Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên - Đại học Quốc gia TP.HCM

Dự án cài đặt Convolutional Autoencoder với CUDA để:
1. Extract features từ CIFAR-10 dataset (60K ảnh 32×32×3)
2. Train SVM classifier trên features đã extract
3. Tối ưu hóa với GPU để đạt speedup >20×

---

## 2. Kiến trúc Autoencoder:

**Tổng quan:**
```
INPUT: (32, 32, 3) → ENCODER (compress) → Latent: (8, 8, 128) = 8,192 
features → DECODER (reconstruct) → OUTPUT: (32, 32, 3) 
```

**Encoder:**
```
Input (32×32×3) → Conv2D(256) + ReLU → MaxPool → (16×16×256)
                → Conv2D(128) + ReLU → MaxPool → (8×8×128) = Latent (8192-D)
```

**Decoder:**
```
Latent (8×8×128) → Conv2D(128) + ReLU → Upsample → (16×16×128)
                 → Conv2D(256) + ReLU → Upsample → (32×32×256)
                 → Conv2D(3) → Output (32×32×3)
```

---

## 3. Cấu trúc thư mục đầy đủ:

```
ParallelProgramming_FinalProject/
├── README.md                    		# ← File này
├── Project_Report.ipynb               	# Notebook để thực thi code
├── CSC14120_2025_Final Project.pdf 	# Mô tả yêu cầu đồ án cuối kỳ
│
├── build/                       		# Thư mục chứa các file được biên dịch
│   ├── phase1                   		# CPU baseline
│   ├── phase2                   		# Naive GPU
│   ├── phase3_1                 		# Optimized 1 GPU
│   ├── phase3_2                 		# Optimized 2 GPU
│   ├── phase4_extract_features  		# Extract feature for SVM
│   └── ...
│
├── cifar-10-batches-bin/        		# CIFAR-10 dataset (binary)
│   ├── data_batch_1.bin
│   ├── data_batch_2.bin
│   ├── data_batch_3.bin
│   ├── data_batch_4.bin
│   ├── data_batch_5.bin
│   ├── test_batch.bin
│   └── ...
│
├── cuda/                         		# CUDA kernels
│   ├── gpu_kernels_naive.h
│   ├── gpu_kernels_naive.cu
│   ├── gpu_kernels_optimized_1.h
│   ├── gpu_kernels_optimized_1.cu
│   ├── gpu_kernels_optimized_2.h
│   └── gpu_kernels_optimized_2.cu
│
├── features/                     		# Thư mục chứa các feature được trích xuất
│   ├── cpu
│   │   ├── train_features.bin
│   │   └── test_features.bin
│   ├── gpu_naive
│   │   └── ...
│   ├── gpu_optimized_1
│   │   └── ...
│   ├── gpu_optimized_2
│   │   └── ...
│
├── include/                      		# Header files
│   ├── cifar10_loader.h
│   ├── autoencoder_cpu.h
│   ├── autoencoder_gpu.h
│   ├── autoencoder_gpu_optimized_1.h
│   ├── autoencoder_gpu_optimized_2.h
│   └── svm_classifier.h
│
├── src/                          		# Source code
│   ├── cifar10_loader.cu
│   ├── reconstruct.cu
│   ├── autoencoder_cpu.cu
│   ├── autoencoder_gpu.cu
│   ├── autoencoder_gpu_optimized_1.cu
│   ├── autoencoder_gpu_optimized_2.cu
│   ├── main_phase1.cu
│   ├── main_phase2.cu
│   ├── main_phase3_1.cu
│   ├── main_phase3_1.cu
│   ├── main_phase4_extract_features.cu
│   └── main_phase4_svm.py
│
└── weights/                      		# Model weights (autoencoder + SVM)
    ├── autoencoder_cpu.bin
    ├── autoencoder_gpu_naive.bin
    ├── autoencoder_gpu_optimized_1.bin
    ├── autoencoder_gpu_optimized_2.bin
    └── svm_cuml.pkl

```

---

## 4. Thực thi chương trình:

**Lưu ý:** Các lệnh dưới đây được cấu hình và kiểm thử trên môi trường Google Colab với GPU NVIDIA Tesla T4 (Compute Capability 7.5). Khi chạy trên môi trường hoặc GPU khác, cần điều chỉnh lại các tham số biên dịch (ví dụ: -arch=sm_75 với T4) cho phù hợp với compute capability của GPU tương ứng.

### 4.1. Setup môi trường:

#### Yêu cầu:
- **CUDA:** >= 11.0 (check: `nvcc --version`)
- **GPU:** Sử dụng GPU trên Google Colab
  - T4: Thiết lập compute capability == 7.5 (-arch=sm_75)
  - L4: Thiết lập compute capability == 8.6 (-arch=sm_86)
  - A100: Thiết lập compute capability == 8.0 (-arch=sm_80)

#### Cài đặt dependencies: Cần cài đặt thư viện cuML. Tuy nhiên, Google Colab đã tải thư viện này nên không phải thực hiện cài đặt

### Tải CIFAR-10 dataset
```
%cd /content/ParallelProgramming_FinalProject
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
```

### 4.2. Compile các module

```
# File load dữ liệu
!nvcc -c -arch=sm_75 src/cifar10_loader.cu -o build/cifar10_loader.o

# Autoencoder của CPU
!nvcc -c -arch=sm_75 src/autoencoder_cpu.cu -o build/autoencoder_cpu.o

# Autoencoder của GPU
!nvcc -c -arch=sm_75 cuda/gpu_kernels_naive.cu -o build/gpu_kernels_naive.o
!nvcc -c -arch=sm_75 src/autoencoder_gpu.cu -o build/autoencoder_gpu.o

# Autoencoder của GPU phiên bản Memory Optimization
!nvcc -c -arch=sm_75 cuda/gpu_kernels_optimized_1.cu -o build/gpu_kernels_optimized_1.o
!nvcc -c -arch=sm_75 src/autoencoder_gpu_optimized_1.cu -o build/autoencoder_gpu_optimized_1.o

# Autoencoder của GPU phiên bản Kernel-block Optimization
!nvcc -c -arch=sm_75 cuda/gpu_kernels_optimized_2.cu -o build/gpu_kernels_optimized_2.o
!nvcc -c -arch=sm_75 src/autoencoder_gpu_optimized_2.cu -o build/autoencoder_gpu_optimized_2.o

# Biên dịch phase 1: CPU
!nvcc -arch=sm_75 src/main_phase1.cu build/cifar10_loader.o build/autoencoder_cpu.o -o build/phase1

# Biên dịch phase 2: Naive GPU
!nvcc -arch=sm_75 src/main_phase2.cu build/cifar10_loader.o build/autoencoder_gpu.o build/gpu_kernels_naive.o -o build/phase2

# Biên dịch phase 3.1: Optimized 1 GPU
!nvcc -arch=sm_75 src/main_phase3_1.cu build/cifar10_loader.o build/autoencoder_gpu_optimized_1.o build/gpu_kernels_naive.o build/gpu_kernels_optimized_1.o -o build/phase3_1

# Biên dịch phase 3.2: Optimized 2 GPU
!nvcc -arch=sm_75 src/main_phase3_2.cu build/cifar10_loader.o build/autoencoder_gpu_optimized_2.o build/gpu_kernels_naive.o build/gpu_kernels_optimized_2.o -o build/phase3_2

# Biên dịch phase 4_extract_features
!nvcc -arch=sm_75 src/main_phase4_extract_features.cu build/cifar10_loader.o build/autoencoder_cpu.o build/autoencoder_gpu.o build/autoencoder_gpu_optimized_1.o build/autoencoder_gpu_optimized_2.o build/gpu_kernels_naive.o build/gpu_kernels_optimized_1.o build/gpu_kernels_optimized_2.o -o build/phase4_extract_features

# Biên dịch reconstruct
!nvcc -arch=sm_75 src/reconstruct.cu \
    build/cifar10_loader.o \
    build/autoencoder_cpu.o \
    build/autoencoder_gpu.o \
    build/gpu_kernels_naive.o \
    build/autoencoder_gpu_optimized_1.o \
    build/autoencoder_gpu_optimized_2.o \
    build/gpu_kernels_optimized_1.o \
    build/gpu_kernels_optimized_2.o \
    -o build/reconstruct
```

### 4.3. Build các phase

```bash
# Phase 1: CPU Baseline
./build/phase1
!./build/reconstruct cpu 10


# Phase 2: Naive GPU
./build/phase2
!./build/reconstruct gpu_naive 10

# Phase 3.1: Optimized 1 GPU  
!./build/phase3_1
!./build/reconstruct gpu_opt1 10

# Phase 3.2: Optimized 1 GPU  
!./build/phase3_2
!./build/reconstruct gpu_opt2 10

# Phase 4_extract_features
# Có thể đổi thành các giá trị --mode là: cpu, gpu_naive, gpu_optimized_1, gpu_optimized_2
!./build/phase4_extract_features --mode gpu_naive

# Phase 4_SVM: Chạy trong file Project_Report.ipynb trên Google Colab

```

---

## 5. Kết quả đạt được

### 5.1. Hiệu năng huấn luyện Autoencoder:

Dựa trên kết quả huấn luyện Autoencoderqua các phase khác nhau (mỗi phase huấn luyện 10 epoch trên GPU), bảng dưới đây tổng hợp đầy đủ các số liệu thực nghiệm thu được:

| Phase | Training Time (s) | Speedup (vs CPU) | Incremental Speedup | Memory Usage | Key Optimization | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU Baseline** | 2211000s | 1.0$\times$ | - | - | - | Giá trị ước tính |
| **GPU Basic** | 4364.2s | 506.62$\times$ | 506.62$\times$ | 0.6 GB | Parallelization | - |
| **GPU Opt v1** | 4081.53s | 541.71$\times$ | 1.07$\times$ | 0.7 GB | Shared memory | - |
| **GPU Opt v2** | 4134.89s | 534.72$\times$ | 0.987$\times$ | 0.7 GB | Kernel Fusion + Unroll loop | - |

### 5.2. Kết quả phân loại SVM:

Kết quả khi train SVM như sau:
```
=== SUMMARY ===
Training accuracy:        65.53%
Test accuracy:            61.13%

=== Per-Class Accuracy ===
       airplane:  64.00%
     automobile:  71.60%
           bird:  45.90%
            cat:  48.30%
           deer:  52.50%
            dog:  47.90%
           frog:  73.90%
          horse:  64.10%
           ship:  74.90%
          truck:  68.20%
```

--

## 6. Tham khảo

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [cuML SVM Documentation](https://docs.rapids.ai/api/cuml/stable/api/)