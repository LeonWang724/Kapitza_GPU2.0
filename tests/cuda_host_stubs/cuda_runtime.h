// Test stub: declarations only, for clang -x cuda --cuda-host-only -nocudainc.
#pragma once

#include <cstddef>

#if defined(__CUDA__)
// Like nvcc, the check force-includes this header before any other.
#define __CUDACC__ 1
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __forceinline__ inline __attribute__((always_inline))
#include "__clang_cuda_builtin_vars.h"
__device__ double exp(double value);
__device__ double cos(double value);
__device__ double sin(double value);
#else
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#endif

#include "cuComplex.h"

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

typedef struct CUstream_st* cudaStream_t;
enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind {
    cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice,
};
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaDeviceScheduleBlockingSync 0x04

struct cudaDeviceProp {
    char name[256];
    std::size_t totalGlobalMem;
    int major;
    int minor;
};

extern "C" {
int cudaConfigureCall(dim3 grid, dim3 block, std::size_t shared = 0, cudaStream_t stream = 0);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);
}
cudaError_t cudaGetLastError();
cudaError_t cudaSetDevice(int device);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device);
cudaError_t cudaDriverGetVersion(int* version);
cudaError_t cudaRuntimeGetVersion(int* version);
cudaError_t cudaMalloc(void** pointer, std::size_t bytes);
cudaError_t cudaFree(void* pointer);
cudaError_t cudaMemcpy(void* destination, const void* source, std::size_t bytes,
                       cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize();
