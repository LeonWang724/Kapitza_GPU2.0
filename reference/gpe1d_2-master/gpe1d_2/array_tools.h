#pragma once
void init_k_grid_1d(MKL_Complex16* input, double dk, int size);
void print_complex_array(MKL_Complex16* input, int size);
void double_to_complex(double* real, double* imag, MKL_Complex16* output, int size);
void complex_to_double(double* real, double* imag, MKL_Complex16* input, int size);
void init_k_grid_2d(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y);
void init_k_grid_3d(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z);

void init_k_grid_3d_x(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z);
void init_k_grid_3d_y(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z);
void init_k_grid_3d_z(MKL_Complex16* input, double dkx, double dky, double dkz, int size_x, int size_y, int size_z);

void init_k_grid_2d_x(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y);
void init_k_grid_2d_y(MKL_Complex16* input, double dkx, double dky, int size_x, int size_y);

void init_k_grid_1d_x(MKL_Complex16* input, double dk, int size);