// Codex CUDA Port: persistent CUB reduction storage for diagnostic scalars.
#pragma once

#include "device_buffer.cuh"

#include <cstddef>

class GpuReducer {
public:
    explicit GpuReducer(int maximum_count);
    double sum(const double* input, int count);

private:
    int maximum_count_;
    std::size_t temporary_bytes_ = 0;
    DeviceBuffer<unsigned char> temporary_;
    DeviceBuffer<double> result_{1};
};

