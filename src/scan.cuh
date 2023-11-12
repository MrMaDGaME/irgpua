#pragma once

void cpu_exclusive_scan(int* input, int* output, int size);
void exclusive_scan(const int* input, int* output, int size, int bsize);
void test_scan(const int size, const int block_size, const int rd);
int test_scan_main(int argc, char** argv);