#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include "mkl.h"
#include "array_tools.h"
#include "config.h"
#include "file_tools.h"
#include "dynamic_potential_1d.h"


#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
using namespace std;



void Dynamic_potential_1d::init_dynamic_potential_1d(void* ptr, MKL_Complex16* potential) {
	struct config_data* conf = (struct config_data*)ptr;

	int size = conf->POINTS_X;
	int small_size = (int)(conf->POINTS_X / conf->DP_PADDING_FACTOR_X);

	rand_arr = (double*)mkl_malloc(sizeof(double) * small_size, 64);
	phase_arr = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * small_size, 64);
	generic_aa = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * small_size, 64);
	phase_arr_padded = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * size, 64);
	phase_arr_padded_k = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * size, 64);
	phase_change = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * small_size, 64);
	phase_change_padded = (MKL_Complex16*)mkl_malloc(sizeof(MKL_Complex16) * size, 64);

	unsigned int time_seed = time(NULL);
	vslNewStream(&rand_stream, VSL_BRNG_MT19937, time_seed);
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rand_stream, small_size, rand_arr, 0.0, 1.0);

	//phase_arr_padded: 0
	memset(phase_arr_padded, 0, sizeof(MKL_Complex16) * size);
	//generic_a: 0
	memset(generic_aa, 0, sizeof(MKL_Complex16) * small_size);
	//phase_arr: 0
	memset(phase_arr, 0, sizeof(MKL_Complex16) * small_size);
	

	//phase_arr: rand_arr
	vdUnpackI(small_size, rand_arr, (double*)phase_arr, 2);

	MKL_Complex16 comp2pi;
	comp2pi.real = 0;
	comp2pi.imag = 2.0 * M_PI;
	//generic_a: 2.0j * pi * phase_arr
	cblas_zaxpy(small_size, &comp2pi, phase_arr, 1, generic_aa, 1);

	//phase_arr: exp( generic_a )
	vzExp(small_size, generic_aa, phase_arr);

	//phase_arr_padded: 0000... phase_arr ...0000
	memcpy(&phase_arr_padded[(int)(size / 2.0) - (int)(small_size / 2.0)], phase_arr, sizeof(MKL_Complex16) * small_size);

	//double res = cblas_dznrm2(conf->POINTS_X, phase_arr_padded, 1) * cblas_dznrm2(conf->POINTS_X, phase_arr_padded, 1);
	//cout << res << endl;
	
	dft_status = DftiCreateDescriptor(&dsc_dft1, DFTI_DOUBLE, DFTI_COMPLEX, 1, size);
	dft_status = DftiSetValue(dsc_dft1, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	dft_status = DftiCommitDescriptor(dsc_dft1);
	dft_status = DftiSetValue(dsc_dft1, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
	dft_status = DftiCommitDescriptor(dsc_dft1);

	//phase_arr_padded_k: FT{ phase_arr_padded }
	dft_status = DftiComputeForward(dsc_dft1, phase_arr_padded, phase_arr_padded_k);
	cblas_zdscal(size, 1.0 / sqrt(size), phase_arr_padded_k, 1);
	

	//res = cblas_dznrm2(conf->POINTS_X, phase_arr_padded_k, 1) * cblas_dznrm2(conf->POINTS_X, phase_arr_padded_k, 1);
	//cout << res << endl;

	memcpy(potential, phase_arr_padded_k, sizeof(MKL_Complex16) * size);
	vzMulByConj(size, potential, potential, potential);

	cblas_zdscal(size, conf->DP_AMPLITUDE, potential, 1);

}

void Dynamic_potential_1d::update_dynamic_potential_1d(void* ptr, MKL_Complex16* potential) {
	struct config_data* conf = (struct config_data*)ptr;

	int size = conf->POINTS_X;
	int small_size = (int)(conf->POINTS_X / conf->DP_PADDING_FACTOR_X);

	//vslNewStream(&rand_stream, VSL_BRNG_MT19937, RANDO_SEED);
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rand_stream, small_size, rand_arr, 0.0, 1.0);

	memset(phase_change_padded, 0, sizeof(MKL_Complex16) * size);
	//phase_arr_padded: 0
	memset(generic_aa, 0, sizeof(MKL_Complex16) * small_size);
	//generic_a: 0
	memset(phase_change, 0, sizeof(MKL_Complex16) * small_size);

	vdUnpackI(small_size, rand_arr, (double*)phase_change, 2);

	MKL_Complex16 comp2pisqrt;
	comp2pisqrt.real = 0;
	comp2pisqrt.imag = 2.0 * M_PI * sqrt(conf->TIME_STEP / conf->DP_CORRELATION_TIME);
	//generic_a: 2.0j * pi * phase_arr
	cblas_zaxpy(small_size, &comp2pisqrt, phase_change, 1, generic_aa, 1);

	//phase_change: exp( generic_a )
	vzExp(small_size, generic_aa, phase_change);

	//phase_change_padded: 0000... phase_arr ...0000
	memcpy(&phase_change_padded[(int)(size / 2.0) - (int)(small_size / 2.0)], phase_change, sizeof(MKL_Complex16) * small_size);

	//phase_arr_padded: phase_change_padded * phase_arr_paddded
	vzMul(size, phase_arr_padded, phase_change_padded, phase_arr_padded);

	//phase_arr_padded_k: FT{ phase_arr_padded }
	dft_status = DftiComputeForward(dsc_dft1, phase_arr_padded, phase_arr_padded_k);
	cblas_zdscal(size, 1.0 / sqrt(size), phase_arr_padded_k, 1);

	//potential: | phase_arr_padded_k | ** 2
	memcpy(potential, phase_arr_padded_k, sizeof(MKL_Complex16) * size);
	vzMulByConj(size, potential, potential, potential);
	
	cblas_zdscal(size, conf->DP_AMPLITUDE, potential, 1);
}

double Dynamic_potential_1d::mean_dynamic_potential_1d(void* ptr, MKL_Complex16* potential) {
	struct config_data* conf = (struct config_data*)ptr;
	double res;
	res = cblas_dzasum(conf->POINTS_X, potential, 1);
	res = res / conf->POINTS_X;
	return(res);
}

void Dynamic_potential_1d::free_memory_dynamic_potential_1d() {
	mkl_free(rand_arr);
	mkl_free(phase_arr);
	mkl_free(generic_aa);
	mkl_free(phase_arr_padded);
	mkl_free(phase_arr_padded_k);
	mkl_free(phase_change);
	mkl_free(phase_change_padded);
}