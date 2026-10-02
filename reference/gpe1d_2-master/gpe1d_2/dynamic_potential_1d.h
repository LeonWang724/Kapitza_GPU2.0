#pragma once
class Dynamic_potential_1d {
private:
	double* rand_arr;
	MKL_Complex16* phase_arr_padded;
	MKL_Complex16* phase_arr_padded_k;
	VSLStreamStatePtr rand_stream;
	MKL_Complex16* phase_arr;
	MKL_Complex16* generic_aa;
	MKL_Complex16* phase_change;
	MKL_Complex16* phase_change_padded;

	MKL_LONG dft_status;
	DFTI_DESCRIPTOR_HANDLE dsc_dft1 = NULL;
public:
	void init_dynamic_potential_1d(void* ptr, MKL_Complex16* potential);
	void update_dynamic_potential_1d(void* ptr, MKL_Complex16* potential);
	void free_memory_dynamic_potential_1d();
	double mean_dynamic_potential_1d(void* ptr, MKL_Complex16* potential);
};