#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

__device__ void exclusive_scan_kernel(int* predicate, int* scan_result, int size) {
    // Un simple scan exclusif, pour une efficacité accrue, considérez un scan hiérarchique ou l'utilisation de bibliothèques existantes.
    if (threadIdx.x > 0 && threadIdx.x < size) {
        scan_result[threadIdx.x] = predicate[threadIdx.x - 1] + scan_result[threadIdx.x - 1];
    } else {
        scan_result[0] = 0;
    }
}

__device__ void inclusive_scan_kernel(int* predicate, int* scan_result, int size) {
    // Un simple scan inclusif, pour une efficacité accrue, considérez un scan hiérarchique ou l'utilisation de bibliothèques existantes.
    if (threadIdx.x > 0 && threadIdx.x < size) {
        scan_result[threadIdx.x] = predicate[threadIdx.x] + scan_result[threadIdx.x - 1];
    } else {
        scan_result[0] = predicate[0];
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
        exclusive_scan_kernel(predicate, scan_result, size);
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
        inclusive_scan_kernel(histo, histo, 256);
        __syncthreads();
        // Normalize the histogram
        histo[threadIdx.x] = (histo[threadIdx.x] * 255) / size;
        __syncthreads();
        // Apply the histogram to the image
        buffer[i] = histo[buffer[i]];
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector <std::string> filepaths;
    for (const auto &dir_entry: recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector <Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;
#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i) {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
//        fix_image_cpu(images[i]);
        int *buffer;
        size_t width = static_cast<size_t>(images[i].width);
        size_t height = static_cast<size_t>(images[i].height);
        size_t size = width * height;
        cudaMalloc(&buffer, width * sizeof(int) * height);
        cudaMemcpy(buffer, images[i].buffer, width * sizeof(int) * height, cudaMemcpyHostToDevice);
        int *predicate;
        cudaMalloc(&predicate, size * sizeof(int));
        cudaMemset(predicate, 0, size * sizeof(int));
        int *scan_result;
        cudaMalloc(&scan_result, size * sizeof(int));
        int *histo;
        cudaMalloc(&histo, 256 * sizeof(int));
        cudaMemset(histo, 0, 256 * sizeof(int));
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        fix_image_gpu<<<numBlocks, blockSize>>>(buffer, images[i].width * images[i].height, predicate, scan_result, histo);
        cudaDeviceSynchronize();
        cudaMemcpy(images[i].buffer, buffer, width * sizeof(int) * height, cudaMemcpyDeviceToHost);
        cudaFree(buffer);
        cudaFree(predicate);
    }
    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i) {
        auto &image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector <ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images]() mutable {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i) {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }
    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);
    return 0;
}
