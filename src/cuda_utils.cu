#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error("CUDA Runtime Error");
    }
}