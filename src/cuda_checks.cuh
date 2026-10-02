// Codex CUDA Port: checked CUDA and cuFFT API helpers.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <sstream>
#include <stdexcept>
#include <string>

inline void cuda_check_impl(cudaError_t status, const char* expression,
                            const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << "CUDA failure at " << file << ':' << line << " for " << expression
            << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

inline const char* cufft_result_name(cufftResult status) {
    switch (status) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "CUFFT_UNKNOWN_ERROR";
    }
}

inline void cufft_check_impl(cufftResult status, const char* expression,
                             const char* file, int line) {
    if (status == CUFFT_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << "cuFFT failure at " << file << ':' << line << " for " << expression
            << ": " << cufft_result_name(status) << " (" << static_cast<int>(status)
            << ')';
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) \
    cuda_check_impl((expression), #expression, __FILE__, __LINE__)
#define CUFFT_CHECK(expression) \
    cufft_check_impl((expression), #expression, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

