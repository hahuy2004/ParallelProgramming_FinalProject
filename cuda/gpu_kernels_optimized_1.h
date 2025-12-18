#ifndef GPU_KERNELS_OPTIMIZED_1_H
#define GPU_KERNELS_OPTIMIZED_1_H

void launch_conv2d_shared_forward(
    const float* d_input, 
    const float* d_weights, 
    const float* d_bias,
    float* d_output,
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding);

// Helper functions to copy biases to constant memory
void copy_bias_to_constant_memory(
    const float* d_bias, int size, int layer);  // layer: 0-4 for conv1-5

#endif // GPU_KERNELS_OPTIMIZED_1_H
