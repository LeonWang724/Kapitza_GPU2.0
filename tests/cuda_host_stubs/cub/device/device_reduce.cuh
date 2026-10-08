// Test stub: declarations only.
#pragma once

#include <cuda_runtime.h>

namespace cub {
struct DeviceReduce {
    template <typename InputIterator, typename OutputIterator>
    static cudaError_t Sum(void* temporary, std::size_t& temporary_bytes,
                           InputIterator input, OutputIterator output, int count,
                           cudaStream_t stream = 0);
};
}  // namespace cub
