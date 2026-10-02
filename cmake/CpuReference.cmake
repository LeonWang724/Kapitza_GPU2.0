# Codex CUDA Port: optional build wrapper around the untouched MKL CPU source.
# The 1D files in reference/gpe1d_2-master are compiled directly and never edited.
set(MKL_INTERFACE lp64)
set(MKL_LINK dynamic)
set(MKL_THREADING intel_thread)
find_package(MKL CONFIG REQUIRED)

set(GPE_CPU_REFERENCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/gpe1d_2-master/gpe1d_2"
)
add_executable(gpe1d_cpu_reference
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_reference_main.cpp"
    "${GPE_CPU_REFERENCE_DIR}/array_tools.cpp"
    "${GPE_CPU_REFERENCE_DIR}/config.cpp"
    "${GPE_CPU_REFERENCE_DIR}/dynamic_potential_1d.cpp"
    "${GPE_CPU_REFERENCE_DIR}/file_tools.cpp"
    "${GPE_CPU_REFERENCE_DIR}/math_routines_1d.cpp"
)
target_include_directories(gpe1d_cpu_reference PRIVATE "${GPE_CPU_REFERENCE_DIR}")
target_link_libraries(gpe1d_cpu_reference PRIVATE MKL::MKL HDF5::HDF5)
target_compile_definitions(gpe1d_cpu_reference PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
if(MSVC)
    target_compile_options(gpe1d_cpu_reference PRIVATE /W4 /Zc:preprocessor)
endif()
set_target_properties(gpe1d_cpu_reference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
)
