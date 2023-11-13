#include "fix_gpu.cuh"
#include "scan.cuh"

void save_array(int *array, int size, std::string name) {
    std::ofstream file(name);
    for (int i = 0; i < size; i++) {
        file << array[i] << " ";
    }
    file << std::endl;
    file.close();
}

__global__ void inclusiveSumKernel(const int *input, int *output, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        int sum = 0;
        for (int i = 0; i <= tid; i++) {
            sum += input[i];
        }
        output[tid] = sum;
    }
}

__global__ void predicateKernel(const int *input, int *output, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        if (input[tid] != -27) {
            output[tid] = 1;
        }
    }
}

__global__ void scatterKernel(int *buffer, int *predicate, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        // Scatter to the corresponding addresses
        if (buffer[i] != -27)
            buffer[predicate[i]] = buffer[i];
    }
}

__global__ void MapHistoKernel(int *buffer, int *histo, int image_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < image_size) {
        // #2 Apply map to fix pixels
        if (i % 4 == 0)
            buffer[i] += 1;
        else if (i % 4 == 1)
            buffer[i] -= 5;
        else if (i % 4 == 2)
            buffer[i] += 3;
        else if (i % 4 == 3)
            buffer[i] -= 8;

        __syncthreads();
        // #3 Histogram
        atomicAdd(&histo[buffer[i]], 1);
        __syncthreads();
    }
}

__global__ void applyHistoKernel(int *buffer, int *histo, int image_size, int cdf_min) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < image_size) {
        buffer[i] = roundf(((histo[buffer[i]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
    }
}

void fix_image_gpu(Image& to_fix){
    size_t size = static_cast<size_t>(to_fix.size());
    size_t image_size = static_cast<size_t>(to_fix.width * to_fix.height);
    // Variables GPU
    int *buffer;
    cudaMalloc(&buffer, size * sizeof(int));
    cudaMemcpy(buffer, to_fix.buffer, size * sizeof(int), cudaMemcpyHostToDevice);
    int *predicate;
    cudaMalloc(&predicate, size * sizeof(int));
    cudaMemset(predicate, 0, size * sizeof(int));
    int *scan_result;
    cudaMalloc(&scan_result, size * sizeof(int));
    int *histo;
    cudaMalloc(&histo, 256 * sizeof(int));
    cudaMemset(histo, 0, 256 * sizeof(int));
    // Kernel calls
    int blockSize = 1024;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // Build predicate vector
    predicateKernel<<<numBlocks, blockSize>>>(buffer, predicate, size);
    // Compute the exclusive sum of the predicate
    exclusive_scan(predicate, scan_result, size, blockSize, true);
    // Copie de scan_result dans predicate
    cudaMemcpy(predicate, scan_result, size * sizeof(int), cudaMemcpyDeviceToDevice);
    // Scatter to the corresponding addresses
    scatterKernel<<<numBlocks, blockSize>>>(buffer, predicate, size);
    // Apply map, Build histo
    numBlocks = (image_size + blockSize - 1) / blockSize;
    MapHistoKernel<<<numBlocks , blockSize>>>(buffer, histo, image_size);
    // Compute the inclusive sum of the histo
    inclusive_scan(histo, scan_result, 256, blockSize, true);
    // Copie de scan_result dans histo_cpu
    int *histo_cpu = new int[256];
    cudaMemcpy(histo_cpu, scan_result, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // Compute cdf_min
    int cdf_min = 0;
    for (int k = 0; k < 256; k++) {
        if (histo_cpu[k] != 0) {
            cdf_min = histo_cpu[k];
            break;
        }
    }
    // Apply the map transformation of the histogram equalization
    applyHistoKernel<<<numBlocks, blockSize>>>(buffer, scan_result, image_size, cdf_min);
    // Copie de buffer dans images[i].buffer
    cudaMemcpy(to_fix.buffer, buffer, size * sizeof(int), cudaMemcpyDeviceToHost);
    // Compute total
    to_fix.to_sort.total = reduce_gpu(buffer, image_size);
    // Free
    cudaFree(buffer);
    cudaFree(predicate);
    cudaFree(scan_result);
    cudaFree(histo);
    cudaDeviceSynchronize();
}

__global__ void reduceKernel(int *buffer, int *sum, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(sum, buffer[tid]);
    }
}

uint64_t reduce_gpu(int *buffer, int size) {
    int *sum;
    cudaMalloc(&sum, sizeof(int));
    cudaMemset(sum, 0, sizeof(int));
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reduceKernel<<<numBlocks, blockSize>>>(buffer, sum, size);
    int sum_cpu;
    cudaMemcpy(&sum_cpu, sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    return sum_cpu;
}
