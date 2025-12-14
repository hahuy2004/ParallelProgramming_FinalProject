# Makefile for CIFAR-10 Autoencoder Project
# Phase 2: Naive GPU Implementation

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2
NVCCFLAGS = -std=c++11 -O2 -arch=sm_75

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
WEIGHT_DIR = weights

# Targets
PHASE2_TARGET = $(BUILD_DIR)/phase2

# Source files
CIFAR_LOADER_H = $(INC_DIR)/cifar10_loader.h
CIFAR_LOADER_CU = $(SRC_DIR)/cifar10_loader.cu
AUTOENCODER_GPU = $(SRC_DIR)/autoencoder_gpu.cu
AUTOENCODER_GPU_H = $(INC_DIR)/autoencoder_gpu.h
MAIN_PHASE2 = $(SRC_DIR)/main_phase2.cu

.PHONY: all clean phase2 dirs

all: dirs phase2

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(WEIGHT_DIR)

phase2: dirs $(PHASE2_TARGET)

$(PHASE2_TARGET): $(MAIN_PHASE2) $(AUTOENCODER_GPU) $(AUTOENCODER_GPU_H) $(CIFAR_LOADER_H) $(CIFAR_LOADER_CU)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) $(MAIN_PHASE2) $(AUTOENCODER_GPU) $(CIFAR_LOADER_CU) -o $(PHASE2_TARGET)
	@echo "Phase 2 built successfully: $(PHASE2_TARGET)"

clean:
	rm -f $(BUILD_DIR)/*
	@echo "Build directory cleaned"

run_phase2: phase2
	./$(PHASE2_TARGET) cifar-10-batches-bin weights/autoencoder_gpu_naive.weights 5 64 0.001 50000

help:
	@echo "Available targets:"
	@echo "  all         - Build all phases (currently only phase2)"
	@echo "  phase2      - Build phase 2 (Naive GPU)"
	@echo "  clean       - Remove build files"
	@echo "  run_phase2  - Build and run phase 2"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage for phase2:"
	@echo "  ./build/phase2 <data_dir> <weights_path> <epochs> <batch_size> <lr> <max_train>"
	@echo "  Example: ./build/phase2 cifar-10-batches-bin weights/gpu_naive.weights 5 64 0.001 50000"
