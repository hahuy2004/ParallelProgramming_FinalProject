# Makefile for CIFAR-10 Autoencoder Project
# All files compiled with NVCC

# Compiler settings
NVCC = nvcc

# Directories
SRC_DIR = src
CUDA_DIR = cuda
INCLUDE_DIR = include
BUILD_DIR = build
WEIGHTS_DIR = weights
LIBSVM_DIR = third_party/libsvm

# Compiler flags (NVCC for all files)
NVCCFLAGS = -std=c++14 -O3 -use_fast_math -I$(INCLUDE_DIR) -I$(CUDA_DIR) -I$(LIBSVM_DIR)

# CUDA architecture (adjust for your GPU)
# Compute capability: 7.5 (RTX 2080, Tesla T4), 8.0 (A100), 8.6 (RTX 3090)
CUDA_ARCH = -arch=sm_75

# CUDA libraries
CUDA_LIBS = -lcudart

# Create directories
$(shell mkdir -p $(BUILD_DIR) $(WEIGHTS_DIR) results)

# Common object files
COMMON_OBJS = $(BUILD_DIR)/cifar10_loader.o

# CPU objects
CPU_OBJS = $(BUILD_DIR)/autoencoder_cpu.o

# GPU objects
GPU_OBJS = $(BUILD_DIR)/autoencoder_gpu.o $(BUILD_DIR)/gpu_kernels.o

# SVM objects
SVM_OBJS = $(BUILD_DIR)/svm_classifier.o $(BUILD_DIR)/svm.o

# Targets
.PHONY: all clean phase1 phase2 phase3 phase4

all: phase1 phase2 phase3 phase4

# Phase 1: CPU Baseline
phase1: $(BUILD_DIR)/phase1

$(BUILD_DIR)/phase1: $(SRC_DIR)/main_phase1.cu $(COMMON_OBJS) $(CPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^

# Phase 2: Naive GPU
phase2: $(BUILD_DIR)/phase2

$(BUILD_DIR)/phase2: $(SRC_DIR)/main_phase2.cu $(COMMON_OBJS) $(GPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(CUDA_LIBS)

# Phase 3: Optimized GPU
phase3: $(BUILD_DIR)/phase3

$(BUILD_DIR)/phase3: $(SRC_DIR)/main_phase3.cu $(COMMON_OBJS) $(BUILD_DIR)/autoencoder_gpu_optimized_1.o $(BUILD_DIR)/autoencoder_gpu_optimized_2.o $(BUILD_DIR)/gpu_kernels.o $(BUILD_DIR)/gpu_kernels_optimized_1.o $(BUILD_DIR)/gpu_kernels_optimized_2.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(CUDA_LIBS)

# Phase 4: Full Pipeline
phase4: $(BUILD_DIR)/phase4

$(BUILD_DIR)/phase4: $(SRC_DIR)/main_phase4.cu $(COMMON_OBJS) $(GPU_OBJS) $(SVM_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $^ $(CUDA_LIBS)

# All object files compiled with NVCC
$(BUILD_DIR)/cifar10_loader.o: $(SRC_DIR)/cifar10_loader.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/autoencoder_cpu.o: $(SRC_DIR)/autoencoder_cpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/autoencoder_gpu.o: $(SRC_DIR)/autoencoder_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/gpu_kernels.o: $(CUDA_DIR)/gpu_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/autoencoder_gpu_optimized_1.o: $(SRC_DIR)/autoencoder_gpu_optimized_1.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/gpu_kernels_optimized_1.o: $(CUDA_DIR)/gpu_kernels_optimized_1.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/autoencoder_gpu_optimized_2.o: $(SRC_DIR)/autoencoder_gpu_optimized_2.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/gpu_kernels_optimized_2.o: $(CUDA_DIR)/gpu_kernels_optimized_2.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/svm_classifier.o: $(SRC_DIR)/svm_classifier.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

$(BUILD_DIR)/svm.o: $(LIBSVM_DIR)/svm.cpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c -o $@ $<

# Run targets
run-phase1: phase1
	./$(BUILD_DIR)/phase1

run-phase2: phase2
	./$(BUILD_DIR)/phase2

run-phase3: phase3
	./$(BUILD_DIR)/phase3

run-phase4: phase4
	./$(BUILD_DIR)/phase4

# Clean
clean:
	rm -rf $(BUILD_DIR)/*.o
	rm -rf $(BUILD_DIR)/phase1 $(BUILD_DIR)/phase2 $(BUILD_DIR)/phase3 $(BUILD_DIR)/phase4

clean-all: clean
	rm -rf $(WEIGHTS_DIR)/*
	rm -rf results/*

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build all phases"
	@echo "  phase1      - Build Phase 1 (CPU baseline)"
	@echo "  phase2      - Build Phase 2 (Naive GPU)"
	@echo "  phase4      - Build Phase 4 (Full pipeline with SVM)"
	@echo "  run-phase1  - Build and run Phase 1"
	@echo "  run-phase2  - Build and run Phase 2"
	@echo "  run-phase4  - Build and run Phase 4"
	@echo "  clean       - Remove build artifacts"
	@echo "  clean-all   - Remove build artifacts and generated files"
	@echo ""
	@echo "Note: Adjust CUDA_ARCH in Makefile for your GPU"
