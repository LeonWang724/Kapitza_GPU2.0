// Test stub: the toolkit's double-precision complex arithmetic, verbatim.
#pragma once

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

struct double2 {
    double x;
    double y;
};
typedef double2 cuDoubleComplex;

__host__ __device__ static inline double cuCreal(cuDoubleComplex x) { return x.x; }
__host__ __device__ static inline double cuCimag(cuDoubleComplex x) { return x.y; }
__host__ __device__ static inline cuDoubleComplex make_cuDoubleComplex(double r, double i) {
    cuDoubleComplex res;
    res.x = r;
    res.y = i;
    return res;
}
__host__ __device__ static inline cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y) {
    return make_cuDoubleComplex(cuCreal(x) + cuCreal(y), cuCimag(x) + cuCimag(y));
}
__host__ __device__ static inline cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y) {
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex((cuCreal(x) * cuCreal(y)) - (cuCimag(x) * cuCimag(y)),
                                (cuCreal(x) * cuCimag(y)) + (cuCimag(x) * cuCreal(y)));
    return prod;
}
