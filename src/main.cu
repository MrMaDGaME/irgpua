#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include <cuda_runtime.h>


//this is new
#include "fix_gpu.cuh" 


#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

// Déclaration de la fonction qui sera définie plus tard.
__global__ void fix_image_gpu(int* buffer, int width, int height);

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    // this is new
    int* d_imageBuffer;
    size_t bufferSize; // cela sera utilisé pour connaître la taille de l'image en termes de mémoire
    //

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        
        Image& currentImage = pipeline.get_image(i); 
        bufferSize = currentImage.width * currentImage.height * sizeof(int); 

        // Allocation de mémoire sur le GPU et copie de l'image sur le GPU
        cudaMalloc(&d_imageBuffer, bufferSize);
        cudaMemcpy(d_imageBuffer, currentImage.buffer, bufferSize, cudaMemcpyHostToDevice);

        // Lancement du traitement sur le GPU
        int numThreads = 256;
        int numBlocks = (currentImage.width * currentImage.height + numThreads - 1) / numThreads;
        fix_image_gpu<<<numBlocks, numThreads>>>(d_imageBuffer, currentImage.width, currentImage.height);
        
        // Copie de l'image traitée du GPU vers le CPU
        cudaMemcpy(currentImage.buffer, d_imageBuffer, bufferSize, cudaMemcpyDeviceToHost);

        // Libération de l'espace mémoire sur le GPU
        cudaFree(d_imageBuffer);

        images[i] = currentImage;

    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
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
