#pragma once

// Kernel
__global__ void exclusive_scan_kernel(const int* input, int* output, int size);
__global__ void inclusive_scan_kernel(const int *input, int *output, int size);

// Scan
void inclusive_scan(int* input, int* output, int size, int bsize, bool allocated);
void exclusive_scan(int* input, int* output, int size, int bsize, bool allocated);

// Test function
int test_scan_main(int argc, char** argv);