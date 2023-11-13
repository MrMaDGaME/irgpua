#include <iostream>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.cuh"
#include "colors.hh"

///////////////////
/* Annex kernels */
///////////////////


__global__ void exclusive_compute_block_sums(const int* input, const int* scan_result, int* block_sums, int size, int blockSize) {
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int index = blockId * blockSize + tid;

    int prev_block_sum_id = ((blockId + 1) * blockSize) - 1;

    if (tid == 0 && index < size) {
        if (blockId == 0) {
            block_sums[blockId] = scan_result[blockSize - 1] + input[blockSize - 1];
        } else {
            block_sums[blockId] = scan_result[prev_block_sum_id] + input[prev_block_sum_id];
        }
    }
}

__global__ void inclusive_compute_block_sums(const int* input, const int* scan_result, int* block_sums, int size, int blockSize) {
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int index = blockId * blockSize + tid;

    int block_sum_id = ((blockId + 1) * blockSize) - 1;

    if (tid == 0 && index < size) {
        block_sums[blockId] = scan_result[block_sum_id];
    }
}

__global__ void adjust_scan_results(int* output, const int* block_sums, int size, int blockSize) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < size && blockIdx.x > 0) {
        output[globalIdx] += block_sums[blockIdx.x - 1];
    }
}


//////////////////
/* Scan kernels */
//////////////////

__global__ void exclusive_scan_kernel(const int* input, int* output, int size) {
    // On effectue le scan dans la shared memory
    extern __shared__ int temp[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    temp[localIdx] = (tid < size) ? input[tid] : 0;
    __syncthreads();

    // On effectue le scan dans la shared memory
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int val = (index - stride >= 0) ? temp[index - stride] : 0;
            temp[index] += val;
        }
        __syncthreads();
    }

    // On applique le "Down-sweep"
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    if (tid < size) {
        output[tid] = (localIdx > 0) ? temp[localIdx - 1] : 0;
    }
}

__global__ void inclusive_scan_kernel(const int* input, int* output, int size) {
    // On crée un tableau dans la mémoire partagée
    extern __shared__ int temp[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    temp[localIdx] = (tid < size) ? input[tid] : 0;
    __syncthreads();

    // On effectue le scan dans la shared memory
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int val = (index - stride >= 0) ? temp[index - stride] : 0;
            temp[index] += val;
        }
        __syncthreads();
    }

    // On applique le "Down-sweep"
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (localIdx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
    }
    __syncthreads();

    if (tid < size) {
        output[tid] = temp[localIdx];
    }
}

////////////////////
/* Scan functions */
////////////////////

// boolean allocated to avoid reallocation
void inclusive_scan(int* input, int* output, int size, int bsize, bool allocated) {
    int blockSize = bsize;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    int* d_input, *d_output, *d_block_sums;
    
    if (allocated) {
        d_input = input;
        d_output = output;
    } else {
        cudaMalloc(&d_input, size * sizeof(int));
        cudaMalloc(&d_output, size * sizeof(int));
        cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    }

    if (bsize >= size) {
        inclusive_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, size);
        cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
            if (!allocated) {
            cudaFree(d_input);
            cudaFree(d_output);
        }
        return;
    }

    cudaMalloc(&d_block_sums, gridSize * sizeof(int));

    inclusive_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, size);
    inclusive_compute_block_sums<<<gridSize, blockSize>>>(d_input, d_output, d_block_sums, size, blockSize);
    
    int* block_sums = new int[gridSize];
    cudaMemcpy(block_sums, d_block_sums, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < gridSize; i++) {
        block_sums[i + 1] += block_sums[i];
    }
    cudaMemcpy(d_block_sums, block_sums, gridSize * sizeof(int), cudaMemcpyHostToDevice);

    adjust_scan_results<<<gridSize, blockSize>>>(d_output, d_block_sums, size, blockSize);

    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    if (!allocated) {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    cudaFree(d_block_sums);
    delete[] block_sums;
}

void exclusive_scan(int* input, int* output, int size, int bsize, bool allocated) {
    int blockSize = bsize;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    int* d_input, *d_output, *d_block_sums;
    
    if (allocated) {
        d_input = input;
        d_output = output;
    } else {
        cudaMalloc(&d_input, size * sizeof(int));
        cudaMalloc(&d_output, size * sizeof(int));
        cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    }

    if (bsize >= size) {
        exclusive_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, size);
        cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
            if (!allocated) {
            cudaFree(d_input);
            cudaFree(d_output);
        }
        return;
    }

    cudaMalloc(&d_block_sums, gridSize * sizeof(int));

    exclusive_scan_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, size);
    exclusive_compute_block_sums<<<gridSize, blockSize>>>(d_input, d_output, d_block_sums, size, blockSize);
    inclusive_scan(d_block_sums, d_block_sums, gridSize, bsize, true);
    adjust_scan_results<<<gridSize, blockSize>>>(d_output, d_block_sums, size, blockSize);

    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    if (!allocated) {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    cudaFree(d_block_sums);
}


///////////////////////////
/* CPU Scans (for debug) */
///////////////////////////


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

void cpu_inclusive_scan(int* input, int* output, int size) {
    std::vector<int> vec(input, input + size);
    for (int i = 0; i < size; i++) {
        vec[i] = input[i];
    }
    std::inclusive_scan(vec.begin(), vec.end(), vec.begin());
    for (int i = 0; i < size; i++) {
        output[i] = vec[i];
    }
}


////////////////////
/* Test functions */
////////////////////

void test_exclusive_scan(const int size, const int block_size, const int rd) {
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
        } else if (rd == 3) {
            srand(time(NULL));
            for (int i = 0; i < size; i++) {
                input[i] = rand() % 16;
            }
        } else {
            srand(42);
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
    exclusive_scan(input, gpu_output, size, block_size, false);

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

void test_inclusive_scan(const int size, const int block_size, const int rd) {
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
        } else if (rd == 3) {
            srand(time(NULL));
            for (int i = 0; i < size; i++) {
                input[i] = rand() % 16;
            }
        } else {
            srand(42);
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
    
    cpu_inclusive_scan(input, cpu_output, size);
    inclusive_scan(input, gpu_output, size, block_size, false);

    for (int i = 0; i < size; i++) {
        output_diff[i] = gpu_output[i] - cpu_output[i];
    }

    std::cout << "CPU Inclusive scan:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << FGRN(cpu_output[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "GPU Inclusive scan:" << std::endl;
    for (int i = 0; i < size; ++i) {
        if (gpu_output[i] == cpu_output[i]) {
            std::cout << FGRN(gpu_output[i]) << " ";
        } else {
            std::cout << FRED(gpu_output[i]) << " ";
        }
    }
    std::cout << std::endl;

    std::cout << "(GPU Inclusive scan - CPU Inclusive scan):" << std::endl;
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
    int rd = 4;
    if (argc >= 2) {
        s = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        bs = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        rd = std::stoi(argv[3]);
    }
    
    std::cout << FBLU("Usage: ./main [ARRAY SIZE] [BLOCK SIZE] [FILL TYPE: 0 -> full zero | 1 -> full one | 2 -> (index + 1) | 3 -> random | 4 -> fixed random]") << std::endl << std::endl;
    test_exclusive_scan(s, bs, rd);
    std::cout << "\n\n\n";
    test_inclusive_scan(s, bs, rd);

    return 0;
}

// __global__ void kernel_inclusive_scan(int *block_sums, int gridSize) {
//     int tid = threadIdx.x;
//     int offset = 1;

//     // Perform an in-place inclusive scan
//     for (int d = gridSize >> 1; d > 0; d >>= 1) { // Build sum in place up the tree
//         __syncthreads();
//         if (tid < d) {
//             int ai = offset*(2*tid+1)-1;
//             int bi = offset*(2*tid+2)-1;

//             block_sums[bi] += block_sums[ai];
//         }
//         offset *= 2;
//     }

//     if (tid == 0) { // Clear the last element
//         block_sums[gridSize - 1] = 0;
//     }

//     // Traverse down tree & build scan
//     for (int d = 1; d < gridSize; d *= 2) {
//         offset >>= 1;
//         __syncthreads();
//         if (tid < d) {
//             int ai = offset*(2*tid+1)-1;
//             int bi = offset*(2*tid+2)-1;

//             int t = block_sums[ai];
//             block_sums[ai] = block_sums[bi];
//             block_sums[bi] += t;
//         }
//     }
//     __syncthreads();
// }

// __global__ void basic_inclusive_scan_kernel(const int *input, int *output, int size) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < size) {
//         int sum = 0;
//         for (int i = 0; i <= tid; i++) {
//             sum += input[i];
//         }
//         output[tid] = sum;
//     }
// }

// __global__ void inclusive_scan_kerneml(const int* input, int* output, int size) {
//     // on crée un tableau dans la mémoire partagé
//     extern __shared__ int temp[];

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int localIdx = threadIdx.x;

//     temp[localIdx] = (tid < size) ? input[tid] : 0;
//     __syncthreads();

//     // scan
//     for (int stride = 1; stride <= blockDim.x; stride *= 2) {
//         int index = (localIdx + 1) * stride * 2 - 1;
//         if (index < blockDim.x) {
//             int val = (index - stride >= 0) ? temp[index - stride] : 0;
//             temp[index] += val;
//         }
//         __syncthreads();
//     }

//     // on effectue le "Down-sweep"
//     for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//         __syncthreads();
//         int index = (localIdx + 1) * stride * 2 - 1;
//         if (index + stride < blockDim.x) {
//             temp[index + stride] += temp[index];
//         }
//     }
//     __syncthreads();

//     if (tid < size) {
//         output[tid] = temp[localIdx];
//     }
// }

// // WORK IN PROGRESS
// void inclusive_scan(int *input, int *output, int size, int bsize) {
//     int *d_input, *d_output;

//     cudaMalloc(&d_input, size * sizeof(int));
//     cudaMalloc(&d_output, size * sizeof(int));

//     cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemset(d_output, 0, size * sizeof(int));

//     // Define block and grid sizes
//     int blockSize = bsize; // Define as per your GPU's architecture
//     int gridSize = (size + blockSize - 1) / blockSize;
//     inclusive_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output, size * sizeof(int));

//     cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

//     cudaFree(d_input);
//     cudaFree(d_output);
// }