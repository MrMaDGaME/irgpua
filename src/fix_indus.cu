#include "fix_indus.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

#define garbage_val -27

void fix_image_indus(Image &to_fix) {
    size_t size = static_cast<size_t>(to_fix.size());
    size_t image_size = static_cast<size_t>(to_fix.width * to_fix.height);
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + size);
    thrust::device_vector<int> d_predicate(size, 0);
    thrust::device_vector<int> d_histo(static_cast<size_t>(256), 0);

    // #1 Compute the predicate
    thrust::transform(d_buffer.begin(), d_buffer.end(), d_predicate.begin(), thrust::placeholders::_1 != garbage_val);

    // Compute the exclusive scan of the predicate
    thrust::exclusive_scan(d_predicate.begin(), d_predicate.end(), d_predicate.begin(), 0);

    // Scatter to the corresponding addresses
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    scatterKernel<<<numBlocks, blockSize>>>(d_buffer.data().get(), d_predicate.data().get(), size);

    // #2 Apply map to fix pixels
    numBlocks = (image_size + blockSize - 1) / blockSize;
    MapHistoKernel<<<numBlocks, blockSize>>>(d_buffer.data().get(), d_histo.data().get(), image_size);

    thrust::inclusive_scan(d_histo.begin(), d_histo.end(), d_histo.begin());

    // Find the first non-zero value in the cumulative histogram
    auto first_none_zero = thrust::find_if(thrust::device, d_histo.begin(), d_histo.end(), [] __device__ (auto v) { return v != 0; });
    const int cdf_min = *first_none_zero;


    // Apply the map transformation of the histogram equalization
    applyHistoKernel<<<numBlocks, blockSize>>>(d_buffer.data().get(), d_histo.data().get(), image_size, cdf_min);

    // Copy the result back to to_fix.data
    thrust::copy(d_buffer.begin(), d_buffer.end(), to_fix.buffer);

    to_fix.to_sort.total = thrust::reduce(d_buffer.begin(), d_buffer.end(), 0);
}
