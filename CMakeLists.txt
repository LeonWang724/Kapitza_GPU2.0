# Codex CUDA Port: native CUDA build for the gpe1d_2 1D solver.
cmake_minimum_required(VERSION 3.27)

project(gpe1d_cuda VERSION 4.0.0 LANGUAGES C CXX CUDA)

option(GPE_CUDA_STRICT_FP "Disable fused multiply-add for closer CPU traceability" ON)
option(GPE_BUILD_CPU_REFERENCE "Build the untouched MKL CPU source as an optional reference" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

execute_process(
    COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE GPE_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GPE_GIT_COMMIT)
    set(GPE_GIT_COMMIT "unavailable")
endif()
execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE GPE_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(GPE_GIT_STATUS)
    set(GPE_GIT_DIRTY "true")
else()
    set(GPE_GIT_DIRTY "false")
endif()
string(TIMESTAMP GPE_BUILD_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)
configure_file(src/build_info.h.in generated/build_info.h @ONLY)

add_executable(gpe1d_cuda
    src/main.cpp
    src/config.cpp
    src/hdf5_io.cpp
    src/kernels_1d.cu
    src/reductions.cu
    src/diagnostics_1d.cu
    src/fused_kernels_1d.cu
    src/solver_1d.cu
)

target_include_directories(gpe1d_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(gpe1d_cuda PRIVATE
    CUDA::cudart
    CUDA::cufft
    HDF5::HDF5
)
target_compile_definitions(gpe1d_cuda PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
)
if(MSVC)
    target_compile_options(gpe1d_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        # CUDA 13's CCCL headers require MSVC's standards-conforming preprocessor.
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
else()
    target_compile_options(gpe1d_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()
if(GPE_CUDA_STRICT_FP)
    target_compile_options(gpe1d_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true>
    )
endif()
set_target_properties(gpe1d_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
)

if(GPE_BUILD_CPU_REFERENCE)
    include(cmake/CpuReference.cmake)
endif()

install(TARGETS gpe1d_cuda RUNTIME DESTINATION bin)
