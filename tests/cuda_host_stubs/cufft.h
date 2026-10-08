// Test stub: declarations only.
#pragma once

#include "cuComplex.h"

typedef int cufftHandle;
typedef cuDoubleComplex cufftDoubleComplex;
enum cufftResult {
    CUFFT_SUCCESS, CUFFT_INVALID_PLAN, CUFFT_ALLOC_FAILED, CUFFT_INVALID_TYPE,
    CUFFT_INVALID_VALUE, CUFFT_INTERNAL_ERROR, CUFFT_EXEC_FAILED, CUFFT_SETUP_FAILED,
    CUFFT_INVALID_SIZE, CUFFT_UNALIGNED_DATA,
};
enum cufftType { CUFFT_Z2Z = 0x69 };
#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);
cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* input,
                         cufftDoubleComplex* output, int direction);
