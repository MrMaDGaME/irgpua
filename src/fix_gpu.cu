#include "fix_gpu.cuh"

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

__global__ void exclusiveSumKernel(const int *input, int *output, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        int sum = 0;
        if (tid > 0) {
            for (int i = 0; i < tid; i++) {
                sum += input[i];
            }
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

__global__ void scatterMapHistoKernel(int *buffer, int *scan_result, int *histo, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        // Scatter to the corresponding addresses
        if (buffer[i] != -27)
            buffer[scan_result[i]] = buffer[i];

        // #2 Apply map to fix pixels
        if (i % 4 == 0)
            buffer[i] += 1;
        else if (i % 4 == 1)
            buffer[i] -= 5;
        else if (i % 4 == 2)
            buffer[i] += 3;
        else if (i % 4 == 3)
            buffer[i] -= 8;

        // #3 Histogram
        __syncthreads();
        atomicAdd(&histo[buffer[i]], 1);
        __syncthreads();
    }
}

__global__ void applyHistoKernel(int *buffer, int *histo, int length, int cdf_min) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        buffer[i] = lroundf(((histo[buffer[i]] - cdf_min) / static_cast<float>(length - cdf_min)) * 255.0f);
    }
}

__global__ void fix_image_gpu(int *buffer, int size, int *predicate, int *scan_result, int *histo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // #1 Compact
        // Build predicate vector
        const int garbage_val = -27;
        if (buffer[i] != garbage_val)
            predicate[i] = 1;

        // Compute the exclusive sum of the predicate
//        exclusive_scan_kernel(predicate, scan_result, size);
        __syncthreads();
        // Scatter to the corresponding addresses
        if (buffer[i] != garbage_val)
            buffer[scan_result[i]] = buffer[i];

        // #2 Apply map to fix pixels
        if (i % 4 == 0)
            buffer[i] += 1;
        else if (i % 4 == 1)
            buffer[i] -= 5;
        else if (i % 4 == 2)
            buffer[i] += 3;
        else if (i % 4 == 3)
            buffer[i] -= 8;

        // #3 Histogram equalization
        __syncthreads();
        atomicAdd(&histo[buffer[i]], 1);
        __syncthreads();
        // Compute the inclusive sum scan of the histogram
//        inclusive_scan_kernel(histo, histo, 256);
        __syncthreads();
        // Normalize the histogram
        histo[threadIdx.x] = (histo[threadIdx.x] * 255) / size;
        __syncthreads();
        // Apply the histogram to the image
        int cdf_min = 0;
        buffer[i] = lroundf(((histo[buffer[i]] - cdf_min) / static_cast<float>(size - cdf_min)) * 255.0f);
        if (buffer[i] < 0 || buffer[i] > 255) {
            printf("bouffe ta soeur\n");
        }
    }
}
