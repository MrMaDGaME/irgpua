#pragma once

#include <iostream>

#include "image.hh"
#include "pipeline.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

void save_array(int *array, int size, std::string name);

__global__ void inclusiveSumKernel(const int *input, int *output, int length);

__global__ void exclusiveSumKernel(const int *input, int *output, int length);

__global__ void predicateKernel(const int *input, int *output, int length);

__global__ void scatterKernel(int *buffer, int *predicate, int length);

__global__ void MapHistoKernel(int *buffer, int *histo, int image_size);

__global__ void applyHistoKernel(int *buffer, int *histo, int image_size, int cdf_min);

void fix_image_gpu(Image& to_fix);

uint64_t reduce_gpu(int *buffer, int size);

__global__ void reduceKernel(int *buffer, int *sum, int size);
