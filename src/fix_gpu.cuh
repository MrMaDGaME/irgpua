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

__global__ void inclusiveSumKernel(const int *input, int *output, int length);

__global__ void exclusiveSumKernel(const int *input, int *output, int length);

__global__ void predicateKernel(const int *input, int *output, int length);

__global__ void scatterKernel(int *buffer, int *predicate, int length);

__global__ void MapHistoKernel(int *buffer, int *histo, int image_size);

__global__ void applyHistoKernel(int *buffer, int *histo, int length, int cdf_min);

void fix_image_gpu(Image& to_fix);
