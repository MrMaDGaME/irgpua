#include <iostream>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.cuh"
#include "colors.hh"

__global__ void adjust_scan_results(int* output, int size, int* block_sums) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < size && blockIdx.x > 0) {
        output[globalIdx] += block_sums[blockIdx.x - 1];
    }
}

__global__ void compute_block_sums(const int* scan_result, int* block_sums, int size, int blockSize) {
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int index = blockId * blockSize + tid;

    int prev_block_sum_id = blockId * blockSize + (blockSize - 1);

    if (tid == 0 && index < size) {  // First thread of each block handles this
        block_sums[blockId] = scan_result[prev_block_sum_id];  // Last element of the block
    }
}

__global__ void compute_block_sums(const int* input, const int* scan_result, int* block_sums, int size, int blockSize) {
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int index = blockId * blockSize + tid;

    int prev_block_sum_id = blockId * blockSize + (blockSize - 1);

    if (tid == 0 && index < size) {  // First thread of each block handles this
        if (blockId == 0) {
            block_sums[blockId] = scan_result[prev_block_sum_id] + input[(blockId + 1) * blockSize * 2];  // Last element of the block
        } else {
            block_sums[blockId] = scan_result[prev_block_sum_id] + input[(blockId + 1) * blockSize];
        }
    }
}

__global__ void exclusive_scan_block_sums(int* block_sums, int num_blocks) {
    extern __shared__ int temp[];  // Shared memory for block sums
    int tid = threadIdx.x;

    // Load block sums into shared memory
    if (tid < num_blocks) {
        temp[tid] = (tid > 0) ? block_sums[tid - 1] : 0;
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Perform scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (tid < num_blocks) {
        block_sums[tid] = temp[tid];
    }
}

__global__ void adjust_scan_results(int* output, const int* block_sums, int size, int blockSize) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < size && blockIdx.x > 0) {
        output[globalIdx] += block_sums[blockIdx.x - 1];
    }
}

__global__ void adjust_scan_results(int* input, int* output, const int* block_sums, int size, int blockSize) {
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int index = blockId * blockSize + tid;

    if (index < size && blockId > 0) {
        // int last_element_in_previous_block_id = (blockId * blockSize);
        output[index] += block_sums[blockId - 1];  // Adjust with the sum of previous blocks
    }
}

__global__ void exclusive_scan_kernel(const int* input, int* output, int size) {
    extern __shared__ int temp[];  // Shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;  // Local index for shared memory

    // Load input into shared memory
    if (tid < size) {
        temp[localIdx] = input[tid];
    } else {
        temp[localIdx] = 0;
    }
    __syncthreads();

    // Perform scan in shared memory
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int val = (index - stride >= 0) ? temp[index - stride] : 0;
            temp[index] += val;
        }
        __syncthreads();
    }

    // Down-sweep phase for exclusive scan
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    // Write results to output array
    if (tid < size) {
        output[tid] = (localIdx > 0) ? temp[localIdx - 1] : 0;
    }
}

void cpu_exclusive_scan(int* input, int* output, int size) {
    std::vector<int> vec(input, input + size);
    for (int i = 0; i < size; i++) {
        vec[i] = input[i];
    }
    std::exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
    for (int i = 0; i < size; i++) {
        output[i] = vec[i];
    }
}


void exclusive_scan(const int* input, int* output, int size, int bsize) {
    int* d_input;
    int* d_output;
    int* d_block_sums;

    // Allocate memory on the device
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    // Define block and grid sizes
    int blockSize = bsize; // Define as per your GPU's architecture
    int gridSize = (size + blockSize - 1) / blockSize;

    // Allocate memory for block sums
    cudaMalloc(&d_block_sums, gridSize * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Run the existing exclusive scan kernel
    exclusive_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, size);

    // Compute block sums
    // Assuming compute_block_sums is implemented as per previous discussion
    compute_block_sums<<<gridSize, blockSize>>>(d_input, d_output, d_block_sums, size, blockSize);

    // Copy block sums back to host for exclusive scan
    int* block_sums = new int[gridSize];
    cudaMemcpy(block_sums, d_block_sums, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Exclusive scan on block sums using GPU kernel
    int blockSumScanBlockSize = 1024;  // Adjust as needed
    int blockSumScanGridSize = (gridSize + blockSumScanBlockSize - 1) / blockSumScanBlockSize;
    exclusive_scan_block_sums<<<blockSumScanGridSize, blockSumScanBlockSize, blockSumScanBlockSize * sizeof(int)>>>(d_block_sums, gridSize);

    // Copy adjusted block sums back to device
    cudaMemcpy(d_block_sums, block_sums, gridSize * sizeof(int), cudaMemcpyHostToDevice);

    // Adjust scan results based on block sums
    // Assuming adjust_scan_results is implemented as per previous discussion
    adjust_scan_results<<<gridSize, blockSize>>>(d_output, d_block_sums, size, blockSize);

    // Copy final results back to host
    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);

    // Free host memory
    delete[] block_sums;
}

void test_scan(const int size, const int block_size, const int rd) {
    std::cout << "BLOCK SIZE: " << FYEL(block_size) << std::endl;
    std::cout << "INPUT SIZE: " << FYEL(size) << std::endl;
    std::cout << std::endl;

    int input[size] = {0};
    if (rd != 0) {
        if (rd == 1) {
            for (int i = 0; i < size; i++) {
                input[i] = 1;
            }
        } else if (rd == 2) {
            for (int i = 0; i < size; i++) {
                input[i] = i + 1;
            }
        } else {
            srand(time(NULL));
            for (int i = 0; i < size; i++) {
                input[i] = rand() % 16;
            }
        }
    }

    std::cout << "Input:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << FYEL(input[i]) << " ";
    }
    std::cout << std::endl;

    int gpu_output[size] = {0};
    int cpu_output[size] = {0};
    int output_diff[size] = {0};
    
    cpu_exclusive_scan(input, cpu_output, size);
    exclusive_scan(input, gpu_output, size, block_size);

    for (int i = 0; i < size; i++) {
        output_diff[i] = gpu_output[i] - cpu_output[i];
    }

    std::cout << "CPU Exclusive scan:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << FGRN(cpu_output[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "GPU Exclusive scan:" << std::endl;
    for (int i = 0; i < size; ++i) {
        if (gpu_output[i] == cpu_output[i]) {
            std::cout << FGRN(gpu_output[i]) << " ";
        } else {
            std::cout << FRED(gpu_output[i]) << " ";
        }
    }
    std::cout << std::endl;

    std::cout << "(GPU Exclusive scan - CPU Exclusive scan):" << std::endl;
    for (int i = 0; i < size; ++i) {
        if (output_diff[i] == 0) {
            std::cout << FGRN(output_diff[i]) << " ";
        } else {
            std::cout << FRED(output_diff[i]) << " ";
        }
    }
    std::cout << std::endl;

    std::cout << std::endl;
    for (int i = 0; i < size; ++i) {
        if (gpu_output[i] != cpu_output[i]) {
            std::cout << "First wrong value at (" << FYEL(i)  << "): " 
            << FRED(gpu_output[i]) << " != " << FGRN(cpu_output[i]) 
            << " | Difference = " << FYEL(gpu_output[i] - cpu_output[i]) 
            << " | Input value at (" << FYEL(i - 1) << ") : " << FYEL(input[i - 1]) 
            << " | Input value at (" << FYEL(i) << ") : " << FYEL(input[i]) 
            << " | Input value at (" << FYEL(i + 1) << ") : " << FYEL(input[i + 1]) 
            << std::endl;
            break;
        }
    }


    std::cout << std::endl;
    int gridSize = (size + block_size - 1) / block_size;
    for (int i = 0; i < gridSize; i++) {
        std::cout << "First element of BLOCK (" << FYEL(i) << "): (GPU) ";
        if (gpu_output[i * block_size] == cpu_output[i * block_size]) {
            std::cout << FGRN(gpu_output[i * block_size]) << " == " << FGRN(cpu_output[i * block_size]) << " (CPU)";
        } else {
            std::cout << FRED(gpu_output[i * block_size]) << " != " << FGRN(cpu_output[i * block_size]) << " (CPU)";
        }
        std::cout << std::endl;
    }
}

int test_scan_main(int argc, char** argv) {
    int s = 8;
    int bs = 8;
    int rd = 3;
    if (argc >= 2) {
        s = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        bs = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        rd = std::stoi(argv[3]);
    }
    
    std::cout << FBLU("Usage: ./main [ARRAY SIZE] [BLOCK SIZE] [FILL TYPE: 0 -> full zero | 1 -> full one | 2 -> (index + 1) | 3 -> random]") << std::endl << std::endl;
    test_scan(s, bs, rd);

    return 0;
}