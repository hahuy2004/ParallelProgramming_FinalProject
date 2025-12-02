cmake_minimum_required(VERSION 3.18)
project(CIFAR10_Autoencoder CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda)
include_directories(${CMAKE_SOURCE_DIR}/third_party/libsvm)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Source files
set(COMMON_SOURCES
    src/cifar10_loader.cpp
)

set(CPU_SOURCES
    src/autoencoder_cpu.cpp
)

set(GPU_SOURCES
    src/autoencoder_gpu.cu
    cuda/gpu_kernels.cu
)

set(SVM_SOURCES
    src/svm_classifier.cpp
    third_party/libsvm/svm.cpp
)

# Phase 1: CPU Baseline
add_executable(phase1
    src/main_phase1.cpp
    ${COMMON_SOURCES}
    ${CPU_SOURCES}
)

# Phase 2: Naive GPU
add_executable(phase2
    src/main_phase2.cpp
    ${COMMON_SOURCES}
    ${GPU_SOURCES}
)

set_target_properties(phase2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Phase 4: Full Pipeline with SVM
add_executable(phase4
    src/main_phase4.cpp
    ${COMMON_SOURCES}
    ${GPU_SOURCES}
    ${SVM_SOURCES}
)

set_target_properties(phase4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create weights and results directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/weights)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
