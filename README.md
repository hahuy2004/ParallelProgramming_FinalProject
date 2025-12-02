# CIFAR-10 Autoencoder + SVM với CUDA

**Đồ án cuối kỳ - Lập trình Song song (CSC14120)**

## 📋 Tổng quan

Dự án implement Convolutional Autoencoder với CUDA để:
1. Extract features từ CIFAR-10 dataset (60K ảnh 32×32×3)
2. Train SVM classifier trên features đã extract
3. Tối ưu hóa với GPU để đạt speedup >20×

### Kiến trúc Autoencoder

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

## 📂 Cấu trúc thư mục

```
Project/
├── README.md                    # ← File này
├── notebook.ipynb               # ← Notebook duy nhất để chạy mọi thứ
├── run_pipeline.py              # Python wrapper
│
├── cifar-10-batches-bin/       # CIFAR-10 dataset (binary)
│   ├── data_batch_1.bin
│   ├── data_batch_2.bin
│   ├── data_batch_3.bin
│   ├── data_batch_4.bin
│   ├── data_batch_5.bin
│   └── test_batch.bin
│
├── include/                     # Header files
│   ├── cifar10_loader.h
│   ├── autoencoder_cpu.h
│   ├── autoencoder_gpu.h
│   ├── autoencoder_gpu_optimized.h
│   └── svm_classifier.h
│
├── src/                        # Source code
│   ├── cifar10_loader.cpp
│   ├── autoencoder_cpu.cpp
│   ├── autoencoder_gpu.cu
│   ├── autoencoder_gpu_optimized.cu
│   ├── svm_classifier.cpp
│   ├── main_phase1.cpp
│   ├── main_phase2.cpp
│   ├── main_phase3.cpp
│   └── main_phase4.cpp
│
├── cuda/                       # CUDA kernels
│   ├── gpu_kernels.h
│   ├── gpu_kernels.cu
│   ├── gpu_kernels_optimized.h
│   └── gpu_kernels_optimized.cu
│
├── Makefile                    # Build system
├── CMakeLists.txt             # Alternative build (CMake)
│
├── build/                      # Compiled binaries (generated)
│   ├── phase1                 # CPU baseline
│   ├── phase2                 # Naive GPU
│   ├── phase3                 # Optimized GPU
│   └── phase4                 # Full pipeline with SVM
│
├── weights/                    # Model weights (generated)
│   ├── autoencoder_cpu.weights
│   ├── autoencoder_gpu.weights
│   └── autoencoder_gpu_optimized.weights
│
└── third_party/               # External libraries
    └── libsvm/                # SVM library
```

---

## 🚀 Quick Start

### 1. Setup môi trường

#### Yêu cầu:
- **CUDA:** >= 11.0 (check: `nvcc --version`)
- **GPU:** NVIDIA với compute capability >= 7.5
- **Compiler:** g++ >= 7.0
- **Python:** >= 3.7 (nếu dùng notebook)

#### Cài đặt dependencies:
```bash
# Clone và build LIBSVM
cd third_party
git clone https://github.com/cjlin1/libsvm.git
cd libsvm
make
cd ../..
```

### 2. Download CIFAR-10 dataset

```bash
# Download
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# Extract
tar -xzf cifar-10-binary.tar.gz

# Đảm bảo có thư mục cifar-10-batches-bin/
```

### 3. Compile project

```bash
# Compile tất cả phases
make all

# Hoặc compile từng phase
make phase1  # CPU baseline
make phase2  # Naive GPU
make phase3  # Optimized GPU
make phase4  # Full pipeline with SVM
```

**Lưu ý:** Điều chỉnh CUDA architecture trong Makefile nếu cần:
```makefile
CUDA_ARCH = -arch=sm_75  # RTX 2080, T4
# sm_80: A100
# sm_86: RTX 3090
```

### 4. Run

```bash
# Phase 1: CPU Baseline
./build/phase1

# Phase 2: Naive GPU
./build/phase2

# Phase 3: Optimized GPU  
./build/phase3

# Phase 4: SVM Classification
./build/phase4
```

---

## 📊 4 Phases Implementation

### Phase 1: CPU Baseline (Sanity Check)
- **File:** `src/autoencoder_cpu.cu`, `src/main_phase1.cu`
- **Mục đích:** Sanity check + Benchmark baseline
- **Configuration:**
  - **Training:** 50,000 images (full dataset), 1 epoch
  - **Purpose:** 
    - ✅ Sanity check: Đảm bảo code không crash, tính toán đúng (no NaN/Inf)
    - ✅ Benchmarking: Đo thời gian để ước lượng full training (20 epochs)
  - **Fast test mode:** Uncomment dòng code để chỉ dùng 300 ảnh test nhanh
- **Features:** 
  - Pure C++ implementation
  - Nested loops cho convolution
  - Simplified backward pass
- **Expected time:** 
  - Full dataset (50,000 images, 1 epoch): ~90 minutes
  - Fast test (300 images, 1 epoch): ~30 seconds

### Phase 2: Naive GPU (Full Training)
- **Files:** `src/autoencoder_gpu.cu`, `cuda/gpu_kernels.cu`
- **Mục đích:** GPU implementation đơn giản
- **Configuration:**
  - **Training:** 50,000 images, 20 epochs (FULL)
  - **Batch size:** 64
- **Features:**
  - Basic CUDA kernels
  - Sequential kernel launches
  - Standard memory transfers
- **Expected speedup:** 6-10× vs CPU

### Phase 3: Optimized GPU (Full Training)
- **Files:** `src/autoencoder_gpu_optimized.cu`, `cuda/gpu_kernels_optimized.cu`
- **Configuration:**
  - **Training:** 50,000 images, 20 epochs (FULL)
  - **Batch size:** 128 (tối ưu GPU utilization)
- **Optimizations:**
  - ✅ **Kernel fusion:** Conv2D + ReLU trong 1 kernel
  - ✅ **Pinned memory:** Faster CPU↔GPU transfers
  - ✅ **Async transfers:** Overlap computation
  - ✅ **Larger batch size:** Better GPU utilization
  - 🔧 **Shared memory tiling:** Template provided (cho future)
- **Expected speedup:** 15-25× vs CPU, ~2× vs Naive GPU

### Phase 4: SVM Classification (Full Pipeline)
- **Files:** `src/main_phase4.cu`, `src/svm_classifier.cu`
- **Configuration:**
  - **Training:** 50,000 images (full dataset)
  - **Test:** 10,000 images
- **Pipeline:**
  1. Load trained autoencoder weights
  2. Extract features (8192-D) từ 60K images
  3. Train SVM với RBF kernel
  4. Evaluate trên test set
- **Expected accuracy:** 60-65%

---

## 🐍 Chạy từ Python/Jupyter

### Setup Python wrapper

```python
from run_pipeline import CIFARAutoencoderPipeline

pipeline = CIFARAutoencoderPipeline()

# Check môi trường
pipeline.check_setup()

# Compile
pipeline.compile_all()

# Run phases
result1 = pipeline.run_phase1_cpu()
result2 = pipeline.run_phase2_gpu()
result3 = pipeline.run_phase3_optimized()
result4 = pipeline.run_phase4_svm()

# Compare
print(f"CPU: {result1['time']:.2f}s")
print(f"Optimized GPU: {result3['time']:.2f}s")
print(f"Speedup: {result1['time'] / result3['time']:.2f}×")
```

### Command line

```bash
python run_pipeline.py check     # Kiểm tra setup
python run_pipeline.py compile   # Compile all
python run_pipeline.py phase3    # Run Phase 3
python run_pipeline.py all       # Run toàn bộ pipeline
python run_pipeline.py profile   # Profile với nsys
```

### Jupyter Notebook

Mở file `notebook.ipynb` - tích hợp đầy đủ:
- Setup & compilation
- Run từng phase
- Visualize kết quả
- So sánh performance
- Report template

---

## 🌐 Google Colab

### Setup trên Colab

```python
# 1. Check GPU
!nvidia-smi

# 2. Clone project
!git clone <your-repo-url>
%cd Project

# 3. Install LIBSVM
!cd third_party && git clone https://github.com/cjlin1/libsvm.git && cd libsvm && make

# 4. Download CIFAR-10
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz

# 5. Compile
!make all

# 6. Run
!./build/phase3
```

### Sử dụng notebook.ipynb trên Colab

1. Upload `notebook.ipynb` lên Google Drive
2. Mở bằng Google Colab
3. Runtime → Change runtime type → GPU (T4)
4. Run từng cell

---

## 🎯 Performance Targets

| Metric | Target | 
|--------|--------|
| Training time | < 10 phút (600s) |
| Feature extraction | < 20s cho 60K images |
| Speedup (Phase 3 vs CPU) | > 20× |
| Test accuracy | 60-65% |

---

## 🎓 Kỹ thuật tối ưu hóa (Phase 3)

### 1. Kernel Fusion
**Problem:** Mỗi kernel có overhead (launch + sync)
**Solution:** Merge Conv2D + ReLU thành 1 kernel

```cpp
// Before: 2 kernels
conv2d_kernel<<<grid, block>>>(input, temp, weights, bias);
relu_kernel<<<grid, block>>>(temp, output);

// After: 1 fused kernel  
conv2d_relu_kernel<<<grid, block>>>(input, output, weights, bias);
```

**Benefit:** Giảm 50% memory traffic, 15-20% faster

### 2. Pinned Memory
**Problem:** Pageable memory → slow CPU↔GPU transfer
**Solution:** Pinned (page-locked) memory

```cpp
// Before
float* h_data = new float[size];
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// After
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // Pinned memory
cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice);
```

**Benefit:** 2× faster transfer

### 3. Async Transfers
**Problem:** Transfer blocking computation
**Solution:** Overlap transfer & computation

```cpp
cudaMemcpyAsync(d_input, h_input, size, ..., stream);
kernel<<<grid, block, 0, stream>>>(...);  // Run while transferring
```

**Benefit:** Hide transfer latency

### 4. Larger Batch Size
**Problem:** Small batches → low GPU utilization
**Solution:** Increase batch size 64 → 128

**Benefit:** Better occupancy, 10-15% faster

---

## 🔧 Troubleshooting

### Compilation errors

**CUDA not found:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Wrong GPU architecture:**
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update Makefile
CUDA_ARCH = -arch=sm_XX  # Replace XX
```

### Runtime errors

**Out of memory:**
- Giảm batch size trong source code
- Phase 2/3: Sửa `int batch_size = 64` → `32`

**LIBSVM not found:**
```bash
cd third_party
git clone https://github.com/cjlin1/libsvm.git
cd libsvm && make
```

**CIFAR-10 dataset not found:**
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
```

---

## 📈 Profiling

### NVIDIA Nsight Systems

```bash
# Profile Phase 3
nsys profile --output=phase3_profile ./build/phase3

# View results
nsys-ui phase3_profile.nsys-rep
```

### NVIDIA Nsight Compute

```bash
# Detailed kernel analysis
ncu --set full --export phase3_kernel ./build/phase3

# View
ncu-ui phase3_kernel.ncu-rep
```

---

## 📝 Report Template

### Nội dung báo cáo

1. **Giới thiệu**
   - Mô tả đề bài
   - Kiến trúc Autoencoder
   - Mục tiêu performance

2. **Implementation**
   - Phase 1: CPU Baseline
   - Phase 2: Naive GPU
   - Phase 3: Optimized GPU (chi tiết optimizations)
   - Phase 4: SVM Classification

3. **Results**
   - Bảng so sánh timing
   - Biểu đồ speedup
   - Confusion matrix
   - Per-class accuracy

4. **Analysis**
   - Profiling results (nsys/ncu)
   - Bottleneck analysis
   - Optimization effectiveness

5. **Conclusion**
   - Thành tựu đạt được
   - Hạn chế
   - Future work

### Video Demo (15-20 phút)

1. Giới thiệu đề bài (2 phút)
2. Demo compilation & execution (3 phút)
3. Giải thích code chính (5 phút)
4. So sánh kết quả (3 phút)
5. Phân tích optimizations (4 phút)
6. Kết luận (1 phút)

---

## 📖 API Reference

### CIFAR10Loader
```cpp
Cifar10Loader loader("cifar-10-batches-bin");
loader.load();  // Load tất cả data

auto& train_images = loader.get_train_images();  // 50000 × 3072
auto& test_images = loader.get_test_images();    // 10000 × 3072
```

### AutoencoderCPU
```cpp
AutoencoderCPU model;
model.train(train_images, num_images, batch_size, epochs, lr);
model.extract_features(images, num_images, features);  // → 8192-D
model.save_weights("path/to/weights");
```

### AutoencoderGPU / AutoencoderGPUOptimized
```cpp
AutoencoderGPU model;  // hoặc AutoencoderGPUOptimized
model.train(train_images, num_images, batch_size, epochs, lr);
model.extract_features(images, num_images, features);
```

### SVMClassifier
```cpp
SVMClassifier svm;
svm.train(train_features, train_labels, num_train);
float accuracy = svm.predict(test_features, test_labels, num_test);
svm.save_model("svm_model.txt");
```

---

## 🤝 Contributing

Nếu muốn mở rộng project:

1. **Phase 3 improvements:**
   - Implement shared memory tiling (template đã có)
   - Multi-stream execution
   - Mixed precision (FP16)

2. **Architecture variants:**
   - Try deeper networks
   - ResNet-style skip connections
   - Different latent dimensions

3. **Other optimizations:**
   - cuDNN library integration
   - Dynamic batch sizing
   - Gradient checkpointing

---

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

## 📄 License

Educational project for CSC14120 - Parallel Programming Course

---

**Good luck! 🚀**
