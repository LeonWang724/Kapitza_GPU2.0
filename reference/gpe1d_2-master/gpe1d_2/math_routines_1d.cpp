#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include "mkl.h"
#include "array_tools.h"
#include "config.h"
#include "file_tools.h"
#include "math_routines_1d.h"
#include "dynamic_potential_1d.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
using namespace std;

void Math_routines_1d::init_1d(void* ptr) {
    struct config_data* conf = (struct config_data*)ptr;
    int TOTAL_POINTS = conf->POINTS_X;
    generic_a = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * TOTAL_POINTS, 64);
    generic_b = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * TOTAL_POINTS, 64);

    dft_deriv = DftiCreateDescriptor(&dsc_dft_deriv, DFTI_DOUBLE, DFTI_COMPLEX, 1, conf->POINTS_X);
    dft_deriv = DftiSetValue(dsc_dft_deriv, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    dft_deriv = DftiCommitDescriptor(dsc_dft_deriv);
    dft_deriv = DftiSetValue(dsc_dft_deriv, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
    dft_deriv = DftiCommitDescriptor(dsc_dft_deriv);

}
/*
Contains the actual fourier step split algorithm. 1d
*/
int Math_routines_1d::start_calculation_1d(void* ptr) {

    struct config_data* conf = (struct config_data*)ptr;

    double STEP_KX;
    double energy_before = 10E100;
    double time_tot = 0.0;
    //Array pointers
    MKL_Complex16* psi_init;
    MKL_Complex16* potential;
    MKL_Complex16* dynamic_potential;
    MKL_Complex16* static_potential;
    MKL_Complex16* k_grid;
    MKL_Complex16* k_propagator;

    double f_flo = 0.0;

    MKL_Complex16* psi_r;
    MKL_Complex16* psi_r2;
    MKL_Complex16* psi_k;
    MKL_Complex16* psi_k2;
    MKL_Complex16* x_propagator;

    Dynamic_potential_1d dyn;
    //Floquet_potential_1d flo;


    create_status_file(&conf->STATUS_FILE[0]);
    status_data status;

    STEP_KX = 2.0 * M_PI / (conf->POINTS_X * conf->STEP_X);

    k_grid = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    psi_init = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    potential = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    dynamic_potential = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    static_potential = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    k_propagator = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);


    psi_r = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    psi_r2 = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    psi_k = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    psi_k2 = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);
    x_propagator = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * conf->POINTS_X, 64);


    if (read_complex_field_file(&conf->POTENTIAL_FILE[0], static_potential, conf->POINTS_X) != 0)
        return(-1);
    if (read_complex_field_file(&conf->INITIAL_FILE[0], psi_init, conf->POINTS_X) != 0)
        return(-1);

    if (conf->USE_FLOQUET_POTENTIAL == true) {
        if (read_complex_field_file(&conf->FLOQUET_POTENTIAL_FILE[0], dynamic_potential, conf->POINTS_X) != 0)
            return(-1);
    }

    double norm;
    double init_norm;

    // write_field_file_1d("psi_init2.h5", generic_a, conf->POINTS_X);

    init_norm = norm_1d(psi_init, conf);
    cout << "Norm of initial state: " << init_norm << endl;

    //cblas_zdscal(conf->POINTS_X, 1.0 / norm, psi_init, 1);



    init_k_grid_1d(k_grid, STEP_KX, conf->POINTS_X);

    MKL_Complex16 two;
    two.real = 2.0;
    two.imag = 0;
    memset(generic_a, 0, sizeof(MKL_Complex16) * conf->POINTS_X);
    vzPowx(conf->POINTS_X, k_grid, two, generic_a);
    //generic_a: k_grid squared
    memset(generic_b, 0, sizeof(MKL_Complex16) * conf->POINTS_X);

    MKL_Complex16 delta_T;
    MKL_Complex16 x_prop_factor_g;
    MKL_Complex16 x_prop_factor;
    x_prop_factor.real = 0;
    x_prop_factor.imag = -0.5 * conf->TIME_STEP;
    if (conf->USE_IMAGINARY_TIME == true) {
        cout << "Using imaginary time." << endl;
        delta_T.real = -0.5 * conf->TIME_STEP;
        delta_T.imag = 0;
        x_prop_factor_g.real = -0.5 * conf->BETA * conf->TIME_STEP;
        x_prop_factor_g.imag = 0;
        x_prop_factor.real = -0.5 * conf->TIME_STEP;
        x_prop_factor.imag = 0;
    }
    else {
        delta_T.real = 0;
        delta_T.imag = -0.5 * conf->TIME_STEP;
        x_prop_factor_g.real = 0;
        x_prop_factor_g.imag = -0.5 * conf->BETA * conf->TIME_STEP;
        x_prop_factor.real = 0;
        x_prop_factor.imag = -0.5 * conf->TIME_STEP;
    }

    if (conf->USE_DYNAMIC_POTENTIAL == true) {
        dyn.init_dynamic_potential_1d(conf, dynamic_potential);
        vzAdd(conf->POINTS_X, static_potential, dynamic_potential, potential);
        //cout << "Mean:" << mean_dynamic_potential_1d(conf, dynamic_potential) << endl;
    }
    else if (conf->USE_FLOQUET_POTENTIAL == true) {
        //flo.init_floquet_potential_1d(conf, dynamic_potential);
        //vzAdd(conf->POINTS_X, static_potential, dynamic_potential, potential); //is this needed?
        //cout << "Mean:" << mean_dynamic_potential_1d(conf, dynamic_potential) << endl;
    }
    else {
        memcpy(potential, static_potential, sizeof(MKL_Complex16) * conf->POINTS_X);
        mkl_free(static_potential);
    }


    cblas_zaxpy(conf->POINTS_X, &delta_T, generic_a, 1, generic_b, 1);
    //generic_b: -I*k^2*dt
    vzExp(conf->POINTS_X, generic_b, k_propagator);

    memcpy(psi_r, psi_init, sizeof(MKL_Complex16) * conf->POINTS_X);

    MKL_LONG dft_status;
    DFTI_DESCRIPTOR_HANDLE dsc_dft1 = NULL;
    dft_status = DftiCreateDescriptor(&dsc_dft1, DFTI_DOUBLE, DFTI_COMPLEX, 1, conf->POINTS_X);
    dft_status = DftiSetValue(dsc_dft1, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    dft_status = DftiCommitDescriptor(dsc_dft1);
    dft_status = DftiSetValue(dsc_dft1, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
    dft_status = DftiCommitDescriptor(dsc_dft1);

    int count;
    double chemical_potential;
    for (count = 0; count < conf->NUM_ITERATIONS; count++) {
        vzMulByConj(conf->POINTS_X, psi_r, psi_r, generic_a);
        //generic_a: psi squared


        if (conf->USE_DYNAMIC_POTENTIAL == true) {
            dyn.update_dynamic_potential_1d(conf, dynamic_potential);
            vzAdd(conf->POINTS_X, static_potential, dynamic_potential, potential);
        }

        else if (conf->USE_FLOQUET_POTENTIAL == true) {
            f_flo = cos(conf->FLOQUET_OMEGA * time_tot);
            cblas_zdscal(conf->POINTS_X, f_flo, dynamic_potential, 1);
            vzAdd(conf->POINTS_X, static_potential, dynamic_potential, potential);
            time_tot += conf->TIME_STEP;
        }

        memset(generic_b, 0, sizeof(MKL_Complex16) * conf->POINTS_X);
        cblas_zaxpy(conf->POINTS_X, &x_prop_factor, potential, 1, generic_b, 1);
        //generic_b: -0.5 * i * dt * V

        cblas_zaxpy(conf->POINTS_X, &x_prop_factor_g, generic_a, 1, generic_b, 1);
        //generic_b: -0.5 * i * dt * V - 0.5*i * dt * g * |psi|^2

        vzExp(conf->POINTS_X, generic_b, x_propagator);
        //x_propagator: exp( generic_b )

        vzMul(conf->POINTS_X, psi_r, x_propagator, psi_r2);
        //psi_r2: psi_r * x_propagator

        dft_status = DftiComputeForward(dsc_dft1, psi_r2, psi_k);
        cblas_zdscal(conf->POINTS_X, 1.0 / conf->POINTS_X, psi_k, 1);
        //psi_k: FT{ psi_r2 }

        vzMul(conf->POINTS_X, psi_k, k_propagator, psi_k2);
        //psi_k2: psi_k * k_propagator

        dft_status = DftiComputeBackward(dsc_dft1, psi_k2, psi_r2);
        //psi_r2 = FT^-1{ psi_k2 }

        vzMulByConj(conf->POINTS_X, psi_r2, psi_r2, generic_a);
        //generic_a: psi_r2 squared

        memset(generic_b, 0, sizeof(MKL_Complex16) * conf->POINTS_X);
        cblas_zaxpy(conf->POINTS_X, &x_prop_factor, potential, 1, generic_b, 1);
        //generic_b: -0.5 * i * dt * V

        cblas_zaxpy(conf->POINTS_X, &x_prop_factor_g, generic_a, 1, generic_b, 1);
        //generic_b: -0.5 * i * dt * V - 0.5*i * dt * g * |psi|^2

        vzExp(conf->POINTS_X, generic_b, x_propagator);
        //x_propagator: exp( generic_b )

        vzMul(conf->POINTS_X, psi_r2, x_propagator, psi_r);
        //psi_r = psi_r2 * x_propagator

        norm = norm_1d(psi_r, conf);
        if (conf->USE_IMAGINARY_TIME == true)
            cblas_zdscal(conf->POINTS_X, sqrt(init_norm / norm), psi_r, 1);
            //psi_r: psi_r / norm

        if (1.0 * count / conf->SAVE_STEP - (int)(1.0 * count / conf->SAVE_STEP) == 0) {
            write_file_to_output_folder_1d(conf->OUTPUT_FOLDER, psi_r, count, conf->POINTS_X);
            if (conf->SAVE_POTENTIAL == true) {
                write_file_to_output_folder_1d(conf->POT_OUTPUT_FOLDER, potential, count, conf->POINTS_X);
            }
        }
        if (1.0 * count / conf->SHOW_STATS_STEP - (int)(1.0 * count / conf->SHOW_STATS_STEP) == 0) {
            norm = norm_1d(psi_r, conf);
            cout << count << " Norm: " << norm << " ";

            status.norm = norm;
            status.iteration = count;

            energy_1d(psi_r, potential, &status, conf);

            cout << "CP: " << status.chemical_pot << " Etot: " << status.total_energy << " Ekin: " << status.kinetic_energy << " Epot: " << status.potential_energy << " Eint: " << status.interaction_energy << "f_flo: " << f_flo;
            if (conf->USE_DYNAMIC_POTENTIAL == true) {
                status.mean_dynamic_potential = dyn.mean_dynamic_potential_1d(conf, dynamic_potential);
                cout << " MDynPot: " << status.mean_dynamic_potential;
            }
            
            append_status_file(&conf->STATUS_FILE[0], &status); 
            if (conf->USE_IMAGINARY_TIME == true) {
                if (energy_before - status.total_energy < conf->STOP_DELTA_ENERGY) {
                    cout << "Energy target met." << endl;
                    write_file_to_output_folder_1d(conf->OUTPUT_FOLDER, psi_r, count, conf->POINTS_X);
                    energy_1d(psi_r, potential, &status, conf);
                    return(0);
                }
                else {
                    cout << " DeltaE: " << energy_before - status.total_energy;
                    energy_before = status.total_energy;
                }
            }
            cout << endl;
        }
    }

    mkl_free(psi_init);
    mkl_free(potential);
    mkl_free(k_grid);
    mkl_free(k_propagator);
    mkl_free(generic_a);
    mkl_free(generic_b);
    mkl_free(psi_r);
    mkl_free(psi_r2);
    mkl_free(psi_k);
    mkl_free(psi_k2);
    mkl_free(x_propagator);
    if (conf->USE_DYNAMIC_POTENTIAL == true) {
        dyn.free_memory_dynamic_potential_1d();
        mkl_free(static_potential);
    }

    mkl_free(dynamic_potential);
    return(0);
}

/*
Calculate the L2 norm of a complex field.
*/
double Math_routines_1d::norm_1d(MKL_Complex16* input, void* ptr) {
    struct config_data* conf = (struct config_data*)ptr;
    int TOTAL_POINTS = conf->POINTS_X;
    double norm = 0;
    norm = cblas_dznrm2(TOTAL_POINTS, input, 1);
    norm = norm * norm;
    norm = norm * conf->STEP_X;
    return(norm);
}


double Math_routines_1d::kinetic_energy_1d(MKL_Complex16* input, void* conf_ptr) {
    struct config_data* conf = (struct config_data*)conf_ptr;
    double STEP_KX;
    double Ekin_x;
    int TOTAL_POINTS = conf->POINTS_X;

    STEP_KX = 2.0 * M_PI / (conf->POINTS_X * conf->STEP_X);

    MKL_Complex16 scaling_factor;
    scaling_factor.real = 1.0 / sqrt(TOTAL_POINTS);
    scaling_factor.imag = 0;
    dft_deriv = DftiComputeForward(dsc_dft_deriv, input, generic_a);
    memset(generic_b, 0, sizeof(MKL_Complex16) * TOTAL_POINTS);
    cblas_zaxpy(TOTAL_POINTS, &scaling_factor, generic_a, 1, generic_b, 1);

    //x
    init_k_grid_1d_x(generic_a, STEP_KX, conf->POINTS_X);
    vzMul(TOTAL_POINTS, generic_b, generic_a, generic_a);
    // dft_deriv2 = DftiComputeBackward(dsc_dft_deriv2, generic_a);
    Ekin_x = 0.5 * norm_1d(generic_a, conf);

    return(Ekin_x);
}

/*
Calculate the energy according to GPE functional.
*/
double Math_routines_1d::energy_1d(MKL_Complex16* input, MKL_Complex16* potential, void* ptr, void* conf_ptr) {
    struct status_data* status = (struct status_data*)ptr;
    struct config_data* conf = (struct config_data*)conf_ptr;
    int i,p;
    double dx_re, dx_im;
    double Ekin = 0;
    double Epot = 0, Eint, Etot;

    int TOTAL_POINTS = conf->POINTS_X;

    Ekin = kinetic_energy_1d(input, conf);

    vzMulByConj(TOTAL_POINTS, input, input, generic_a);
    memset(generic_b, 0, sizeof(MKL_Complex16) * TOTAL_POINTS);
    MKL_Complex16 two;
    two.real = 2.0;
    two.imag = 0;
    vzPowx(TOTAL_POINTS, generic_a, two, generic_b);
    Eint = conf->STEP_X * 0.5 * conf->BETA * cblas_dzasum(TOTAL_POINTS, generic_b, 1);


    vzMulByConj(TOTAL_POINTS, input, input, generic_a);
    memset(generic_b, 0, sizeof(MKL_Complex16) * TOTAL_POINTS);
    vzMul(TOTAL_POINTS, generic_a, potential, generic_b);
    //Epot = cblas_dzasum(TOTAL_POINTS, generic_b, 1);
    for (p = 0; p < conf->POINTS_X; p++) {
        Epot += generic_b[p].real;
    }


    Epot = conf->STEP_X *  Epot;
    Etot = Ekin + Epot + Eint;
    

    status->kinetic_energy = Ekin;
    status->potential_energy = Epot;
    status->interaction_energy = Eint;
    status->total_energy = Etot;
    
    status->chemical_pot = Etot + Eint;

    return(Etot);
}