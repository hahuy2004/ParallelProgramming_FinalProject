# Phase 2: Naive GPU Implementation - Updated

## Thay đổi chính

Code Phase 2 đã được cập nhật theo cấu trúc của FinalCuda để đơn giản và hiệu quả hơn:

### 1. **Cấu trúc mới**
- Sử dụng `train_step()` cho từng image thay vì batch processing phức tạp
- CUDA kernels được tối ưu hóa và dễ đọc hơn
- Loại bỏ các buffer không cần thiết

### 2. **API đơn giản hơn**
- `float train_step(const float* input_chw, float learning_rate)` - Train 1 ảnh
- `bool save_weights(const std::string& filepath)` - Lưu weights
- `bool load_weights(const std::string& filepath)` - Load weights  
- `void extract_features(const float* input_chw, float* output_features)` - Trích xuất features

### 3. **Files chính**
- `include/autoencoder_gpu.h` - Header đã được cập nhật
- `src/autoencoder_gpu.cu` - Implementation mới (dựa trên FinalCuda)
- `src/main_phase2.cu` - Main program đã được đơn giản hóa
- `Makefile` - Build script

## Cách sử dụng

### 1. Build

```bash
make phase2
```

### 2. Run

```bash
# Cú pháp đầy đủ:
./build/phase2 <data_dir> <weights_path> <epochs> <batch_size> <lr> <max_train>

# Ví dụ - Train với full dataset (50000 ảnh, 5 epochs):
./build/phase2 cifar-10-batches-bin weights/autoencoder_gpu_naive.weights 5 64 0.001 50000

# Debug với ít ảnh hơn (1000 ảnh):
./build/phase2 cifar-10-batches-bin weights/autoencoder_gpu_naive.weights 2 64 0.001 1000

# Sử dụng shortcut:
make run_phase2
```

### 3. Tham số

- `data_dir`: Thư mục chứa CIFAR-10 dataset (default: `cifar-10-batches-bin`)
- `weights_path`: Đường dẫn lưu weights (default: `weights/autoencoder_gpu_naive.weights`)
- `epochs`: Số epoch (default: 5)
- `batch_size`: Batch size (default: 64)
- `lr`: Learning rate (default: 0.001)
- `max_train`: Số lượng ảnh tối đa để train (default: 50000)

## So sánh với code cũ

| Aspect | Code cũ | Code mới (FinalCuda style) |
|--------|---------|---------------------------|
| Train API | `train(vector, num, batch, epochs, lr)` | `train_step(float*, lr)` per image |
| Buffer management | Phức tạp với batch buffers | Đơn giản, 1 image buffers |
| CUDA kernels | Sử dụng từ gpu_kernels.cu | Tích hợp trực tiếp trong file |
| Weight init | Xavier/He init | Simple random init |
| Code clarity | Phức tạp hơn | Đơn giản, dễ đọc |

## Kết quả mong đợi

- **Training time**: ~X giây cho 50000 ảnh (tùy GPU)
- **Final loss**: ~0.01-0.05 sau 5 epochs
- **Weights file**: ~3.0 MB (751,875 parameters × 4 bytes)

## Tương thích với Phase tiếp theo

File weights được lưu với format giống FinalCuda, có thể:
- Load vào Phase 3 (Optimized GPU)
- Sử dụng cho feature extraction
- Train tiếp với Phase 4 (SVM)

## Troubleshooting

### Lỗi compile
```bash
make clean
make phase2
```

### CUDA out of memory
Giảm `max_train` xuống:
```bash
./build/phase2 cifar-10-batches-bin weights/gpu.weights 5 64 0.001 1000
```

### Loss không giảm
- Kiểm tra learning rate (thử 0.001 hoặc 0.0001)
- Kiểm tra data đã được normalize chưa
- Kiểm tra CUDA errors

## File backup

File implementation cũ đã được backup tại:
- `src/autoencoder_gpu_old.cu.backup`

Nếu cần rollback:
```bash
mv src/autoencoder_gpu.cu src/autoencoder_gpu_new.cu
mv src/autoencoder_gpu_old.cu.backup src/autoencoder_gpu.cu
```
