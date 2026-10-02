#include "hdf5.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include "mkl.h"
#include "array_tools.h"

using namespace std;

void smaller_into_bigger_array_1d(MKL_Complex16* input, MKL_Complex16* output, int size_smaller, int size_bigger) {
    memcpy(&output[(int)(size_bigger/2.0) - (int)(size_smaller / 2.0)], input, sizeof(MKL_Complex16) * size_smaller);
}

void init_k_grid_1d(MKL_Complex16* input, double dk, int size) {
    int i;
    int p;
    double kx, ky;
    for (p = 0; p < size; p++) {
        if (p >= (int)(size / 2.0))
            kx = (p - size) * dk;
        else
            kx = p * dk;

        input[p].real = kx;
        input[p].imag = 0;
    }
}

void init_k_grid_1d_x(MKL_Complex16* input, double dk, int size) {
    int i;
    int p;
    double kx, ky;
    for (p = 0; p < size; p++) {
        if (p >= (int)(size / 2.0))
            kx = (p - size) * dk;
        else
            kx = p * dk;

        input[p].real = 0;
        input[p].imag = kx;
    }
}

void init_k_grid_2d(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y) {
    int p, q;
    double kx, ky;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            //input[q + size_y * p].real = sqrt(kx * kx + ky * ky);
            input[q + size_y * p].real = sqrt(kx * kx + ky * ky);
            input[q + size_y * p].imag = 0;
        }
    }
}

void init_k_grid_2d_x(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y) {
    int p, q;
    double kx, ky;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            input[q + size_y * p].real = 0;
            input[q + size_y * p].imag = kx;
        }
    }
}

void init_k_grid_2d_y(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y) {
    int p, q;
    double kx, ky;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            input[q + size_y * p].real = 0;
            input[q + size_y * p].imag = ky;
        }
    }
}

void init_k_grid_3d_z(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z) {
    int p, q, r;
    double kx, ky, kz;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            for (r = 0; r < size_z; r++) {
                if (r >= (int)(size_z / 2.0))
                    kz = (r - size_z) * dkz;
                else
                    kz = r * dkz;

                input[r + size_z * (q + size_y * p)].real = 0;
                input[r + size_z * (q + size_y * p)].imag = kz;

            }
        }
    }
}

void init_k_grid_3d_y(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z) {
    int p, q, r;
    double kx, ky, kz;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            for (r = 0; r < size_z; r++) {
                if (r >= (int)(size_z / 2.0))
                    kz = (r - size_z) * dkz;
                else
                    kz = r * dkz;

                input[r + size_z * (q + size_y * p)].real = 0;
                input[r + size_z * (q + size_y * p)].imag = ky;

            }
        }
    }
}

void init_k_grid_3d_x(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z) {
    int p, q, r;
    double kx, ky, kz;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            for (r = 0; r < size_z; r++) {
                if (r >= (int)(size_z / 2.0))
                    kz = (r - size_z) * dkz;
                else
                    kz = r * dkz;

                input[r + size_z * (q + size_y * p)].real = 0;
                input[r + size_z * (q + size_y * p)].imag = kx;

            }
        }
    }
}

void init_k_grid_3d(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z) {
    int p, q, r;
    double kx, ky, kz;
    for (p = 0; p < size_x; p++) {
        if (p >= (int)(size_x / 2.0))
            kx = (p - size_x) * dkx;
        else
            kx = p * dkx;

        for (q = 0; q < size_y; q++) {
            if (q >= (int)(size_y / 2.0))
                ky = (q - size_y) * dky;
            else
                ky = q * dky;

            for (r = 0; r < size_z; r++) {
                if (r >= (int)(size_z / 2.0))
                    kz = (r - size_z) * dkz;
                else
                    kz = r * dkz;

                input[r + size_z * (q + size_y * p)].real = kx * kx + ky * ky + kz * kz;
                input[r + size_z * (q + size_y * p)].imag = 0;

            }
        }
    }
}

void print_complex_array(MKL_Complex16* input, int size) {
    int i;
    for (i = 0; i < size; i++) {
        cout << input[i].real << " " << input[i].imag << "j" << endl;
    }
}

void double_to_complex(double* real, double* imag, MKL_Complex16* output, int size) {
    int i;
    for (i = 0; i < size; i++) {
        output[i].real = real[i];
        output[i].imag = imag[i];
    }
}

void complex_to_double(double* real, double* imag, MKL_Complex16* input, int size) {
    int i;
    for (i = 0; i < size; i++) {
        real[i] = input[i].real;
        imag[i] = input[i].imag;
    }
}

