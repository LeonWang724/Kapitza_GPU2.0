// Codex CUDA Port: reliable double-precision norm and energy reductions using CUB.
#include "reductions.cuh"

#include "cuda_checks.cuh"

#include <cub/device/device_reduce.cuh>
#include <stdexcept>

GpuReducer::GpuReducer(int maximum_count) : maximum_count_(maximum_count) {
    if (maximum_count_ <= 0) {
        throw std::runtime_error("GpuReducer requires a positive maximum count.");
    }
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temporary_bytes_,
                                      static_cast<const double*>(nullptr),
                                      result_.data(), maximum_count_));
    temporary_.allocate(temporary_bytes_);
}

double GpuReducer::sum(const double* input, int count) {
    if (count <= 0 || count > maximum_count_) {
        throw std::runtime_error("Reduction count is outside the allocated range.");
    }
    CUDA_CHECK(cub::DeviceReduce::Sum(temporary_.data(), temporary_bytes_, input,
                                      result_.data(), count));
    double host_result = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_result, result_.data(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    return host_result;
}

