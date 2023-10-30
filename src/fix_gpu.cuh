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

__global__ void scatterMapHistoKernel(int *buffer, int *scan_result, int *histo, int length);

__global__ void applyHistoKernel(int *buffer, int *histo, int length, int cdf_min);

__global__ void fix_image_gpu(int *buffer, int size, int *predicate, int *scan_result, int *histo);
