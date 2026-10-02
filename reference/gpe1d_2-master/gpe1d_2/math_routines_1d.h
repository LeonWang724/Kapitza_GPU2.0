class Math_routines_1d {
private:
	MKL_Complex16* generic_a;
	MKL_Complex16* generic_b;
	MKL_LONG dft_deriv;
	DFTI_DESCRIPTOR_HANDLE dsc_dft_deriv = NULL;
public:
	int start_calculation_1d(void* ptr);
	double energy_1d(MKL_Complex16* input, MKL_Complex16* potential, void* ptr, void* conf_ptr);
	double norm_1d(MKL_Complex16* input, void* ptr);
	void init_1d(void* ptr);
	double kinetic_energy_1d(MKL_Complex16* input, void* conf_ptr);
};
