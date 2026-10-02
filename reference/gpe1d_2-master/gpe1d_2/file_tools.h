#pragma once

#define TYPE_PSI 1
#define TYPE_POT 2
#define TYPE_RED 3

#ifdef __linux__ 
#define DIR_SLASH "/"
#elif _WIN32
#define DIR_SLASH "\\"

#endif

struct status_data {
	int iteration;
	double norm;
	double chemical_pot;
	double total_energy;
	double kinetic_energy;
	double potential_energy;
	double interaction_energy;
	double mean_dynamic_potential;
};

int read_complex_field_file(const char* field_file, MKL_Complex16* input, int size);
void write_field_file_1d(const char* field_file, MKL_Complex16* input, int size);
void write_file_to_output_folder_1d(const char* output_folder, MKL_Complex16* input, int count, int size);
void write_field_file_2d(const char* field_file, MKL_Complex16* input, int size_x, int size_y);
void write_file_to_output_folder_2d(const char* output_folder, MKL_Complex16* input, int count, int size_x, int size_y);
void write_file_to_output_folder_3d(void* ptr, MKL_Complex16* input, int count, int type);
void write_field_file_3d(const char* field_file, MKL_Complex16* input, int size_x, int size_y, int size_z);
void create_status_file(const char* file_name);
void append_status_file(const char* file_name, void* ptr);