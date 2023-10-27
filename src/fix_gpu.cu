#include "image.hh"
#include <cuda_runtime.h>




__global__ void compact_kernel(unsigned char* buffer, int* predicate, int image_size, const int garbage_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size)
    {
        predicate[idx] = (buffer[idx] != garbage_val) ? 1 : 0;
    }
}

__global__ void scatter_kernel(unsigned char* buffer, int* predicate, int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size && buffer[idx] != garbage_val)
    {
        buffer[predicate[idx]] = buffer[idx];
    }
}

__global__ void apply_map_kernel(unsigned char* buffer, int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size)
    {
        if (idx % 4 == 0)
            buffer[idx] += 1;
        else if (idx % 4 == 1)
            buffer[idx] -= 5;
        else if (idx % 4 == 2)
            buffer[idx] += 3;
        else if (idx % 4 == 3)
            buffer[idx] -= 8;
    }
}

__global__ void histogram_kernel(unsigned char* buffer, int* histo, int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size)
    {
        atomicAdd(&histo[buffer[idx]], 1);
    }
}

__global__ void equalize_histogram_kernel(unsigned char* buffer, int* histo, int cdf_min, int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size)
    {
        buffer[idx] = roundf(((histo[buffer[idx]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
    }
}

void fix_image_gpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    unsigned char* d_buffer;
    int* d_predicate;
    int* d_histo;

    cudaMalloc((void**)&d_buffer, sizeof(unsigned char) * image_size);
    cudaMalloc((void**)&d_predicate, sizeof(int) * image_size);
    cudaMalloc((void**)&d_histo, sizeof(int) * 256);

    cudaMemcpy(d_buffer, to_fix.buffer, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);

    // #1 Compact
    compact_kernel<<<(image_size + 255) / 256, 256>>>(d_buffer, d_predicate, image_size, -27);
    // Run an exclusive scan on d_predicate using thrust or implement a kernel.
    scatter_kernel<<<(image_size + 255) / 256, 256>>>(d_buffer, d_predicate, image_size);

    // #2 Apply map to fix pixels
    apply_map_kernel<<<(image_size + 255) / 256, 256>>>(d_buffer, image_size);

    // #3 Histogram equalization
    histogram_kernel<<<(image_size + 255) / 256, 256>>>(d_buffer, d_histo, image_size);

    // Compute the inclusive sum scan of the histogram on d_histo.
    // Find the first non-zero value in the cumulative histogram on the CPU.
    // Apply the map transformation of the histogram equalization.
    equalize_histogram_kernel<<<(image_size + 255) / 256, 256>>>(d_buffer, d_histo, cdf_min, image_size);

    cudaMemcpy(to_fix.buffer, d_buffer, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_buffer);
    cudaFree(d_predicate);
    cudaFree(d_histo);
}
