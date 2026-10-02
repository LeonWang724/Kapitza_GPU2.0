#include "hdf5.h"
#include <iostream>
#include <fstream>
#include <sstream>   
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <memory.h>
#include "mkl.h"
#include "file_tools.h"
#include "array_tools.h"
#include "config.h"

using namespace std;

/*
Creates and writes header of the status csv file.
*/
void create_status_file(const char* file_name) {
    ofstream file;
    file.open(file_name);
    file << "Iteration;Norm;Etot;Ekin;Epot;Eint;Chpot;Meandynpot;" << endl;
    file.close();
}

/*
Appends new line to status file.
*/
void append_status_file(const char* file_name, void* ptr) {
    struct status_data* status = (struct status_data*)ptr;

    ofstream file;
    file.open(file_name, std::ios_base::app);
    file << status->iteration << ";" << status->norm << ";" << status->total_energy << ";" << status->kinetic_energy << ";";
    file << status->potential_energy << ";" << status->interaction_energy << ";" << status->chemical_pot << ";";
    file << status->mean_dynamic_potential << ";" << endl;
    file.close();
}

/*
Reads in a complex field file of HDF5 type.
*/
int read_complex_field_file(const char* field_file, MKL_Complex16* input, int size) {
    double* real;
    double* imag;
    real = (double*)malloc(sizeof(double) * size);
    imag = (double*)malloc(sizeof(double) * size);

    hid_t           file, space, dset, dset2, dcpl;    /* Handles */
    H5Z_filter_t    filter_type;
    size_t          nelmts;
    herr_t          status;
    unsigned int    flags, filter_info;
    int i;

    file = H5Fopen(field_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file, "REAL", H5P_DEFAULT);
    dset2 = H5Dopen(file, "IMAGINARY", H5P_DEFAULT);
    if (dset < 0 || dset2 < 0)
        return(-1);


    status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &real[0]);
    status = H5Dread(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &imag[0]);

    status = H5Dclose(dset);
    status = H5Fclose(file);

    double_to_complex(real, imag, input, size);

    free(real);
    free(imag);

    /*for (i = 0; i < 10; i++) {
        cout << input[i].real << " " << input[i].imag << "j" << endl;
    }*/
    return(0);
}

/*
Writes to a complex field file of HDF5 type.
*/
void write_field_file_3d(const char* field_file, MKL_Complex16* input, int size_x, int size_y, int size_z) {
    double* real;
    double* imag;
    real = (double*)malloc(sizeof(double) * size_x * size_y * size_z);
    imag = (double*)malloc(sizeof(double) * size_x * size_y * size_z);

    complex_to_double(real, imag, input, size_x * size_y * size_z);

    hid_t       file, space, dset, dset_im;          /* Handles */
    herr_t      status;
    hsize_t     dims[3] = { size_x, size_y, size_z };

    file = H5Fcreate(&field_file[0], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    space = H5Screate_simple(3, dims, NULL);

    dset = H5Dcreate(file, "REAL", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    dset_im = H5Dcreate(file, "IMAGINARY", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &real[0]);
    status = H5Dwrite(dset_im, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &imag[0]);

    status = H5Dclose(dset);
    status = H5Dclose(dset_im);
    status = H5Sclose(space);
    status = H5Fclose(file);

    free(real);
    free(imag);

}

/*
Writes to a complex field file of HDF5 type.
*/
void write_field_file_2d(const char* field_file, MKL_Complex16* input, int size_x, int size_y) {
    double* real;
    double* imag;
    real = (double*)malloc(sizeof(double) * size_x * size_y);
    imag = (double*)malloc(sizeof(double) * size_x * size_y);

    complex_to_double(real, imag, input, size_x * size_y);

    hid_t       file, space, dset, dset_im;          /* Handles */
    herr_t      status;
    hsize_t     dims[2] = { size_x, size_y };

    file = H5Fcreate(&field_file[0], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    space = H5Screate_simple(2, dims, NULL);

    dset = H5Dcreate(file, "REAL", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    dset_im = H5Dcreate(file, "IMAGINARY", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &real[0]);
    status = H5Dwrite(dset_im, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &imag[0]);

    status = H5Dclose(dset);
    status = H5Dclose(dset_im);
    status = H5Sclose(space);
    status = H5Fclose(file);

    free(real);
    free(imag);

}

/*
Writes to a complex field file of HDF5 type.
*/
void write_field_file_1d(const char* field_file, MKL_Complex16* input, int size) {
    double* real;
    double* imag;
    real = (double*)malloc(sizeof(double) * size);
    imag = (double*)malloc(sizeof(double) * size);

    complex_to_double(real, imag, input, size);

    hid_t       file, space, dset, dset_im;          /* Handles */
    herr_t      status;
    hsize_t     dims[1] = { size };

    file = H5Fcreate(&field_file[0], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    space = H5Screate_simple(1, dims, NULL);

    dset = H5Dcreate(file, "REAL", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    dset_im = H5Dcreate(file, "IMAGINARY", H5T_IEEE_F64LE, space, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &real[0]);
    status = H5Dwrite(dset_im, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &imag[0]);

    status = H5Dclose(dset);
    status = H5Dclose(dset_im);
    status = H5Sclose(space);
    status = H5Fclose(file);

    free(real);
    free(imag);

}

/*
Sets up file path for easier sorting later.
*/
void write_file_to_output_folder_1d(const char* output_folder, MKL_Complex16* input, int count, int size) {
    string out_file, number_str;
    out_file = strdup(output_folder);
    out_file.append(DIR_SLASH);
    out_file.append("0000000000000000");
    number_str = to_string(count);
    out_file.replace(out_file.length() - number_str.length(), number_str.length(), number_str);
    out_file.append(".h5");
    //cout << out_file << endl;
    write_field_file_1d(&out_file[0], input, size);
}

/*
Sets up file path for easier sorting later.
*/
void write_file_to_output_folder_2d(const char* output_folder, MKL_Complex16* input, int count, int size_x, int size_y) {
    string out_file, number_str;
    out_file = strdup(output_folder);
    out_file.append(DIR_SLASH);
    out_file.append("0000000000000000");
    number_str = to_string(count);
    out_file.replace(out_file.length() - number_str.length(), number_str.length(), number_str);
    out_file.append(".h5");
    //cout << out_file << endl;
    write_field_file_2d(&out_file[0], input, size_x, size_y);
}

/*
Sets up file path for easier sorting later.
*/
void write_file_to_output_folder_3d(void* ptr, MKL_Complex16* input, int count, int type) {
    struct config_data* conf = (struct config_data*)ptr;
    string out_file, number_str;
    out_file = strdup(conf->OUTPUT_FOLDER);
    out_file.append(DIR_SLASH);
    out_file.append("0000000000000000");
    number_str = to_string(count);
    out_file.replace(out_file.length() - number_str.length(), number_str.length(), number_str);

    if (type == TYPE_PSI)
        out_file.append(".psi");
    else if (type == TYPE_POT)
        out_file.append(".pot");
    else if (type == TYPE_RED)
        out_file.append(".red");
    write_field_file_3d(&out_file[0], input, conf->POINTS_X, conf->POINTS_Y, conf->POINTS_Z);
}