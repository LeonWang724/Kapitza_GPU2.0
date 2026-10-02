// Codex CUDA Port: small RAII wrapper for persistent CUDA allocations.
#pragma once

#include "cuda_checks.cuh"

#include <cstddef>
#include <utility>

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
        }
        data_ = nullptr;
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

