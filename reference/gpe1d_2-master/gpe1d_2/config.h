//#include <string>

struct config_data {
    int NUM_THREADS;
    int NUM_ITERATIONS;
    int SAVE_STEP;
    int SHOW_STATS_STEP;
    int dimension;
    const char* POTENTIAL_FILE;
    const char* FLOQUET_POTENTIAL_FILE;
    const char* INITIAL_FILE;
    const char* STATUS_FILE;
    const char* OUTPUT_FOLDER;
    const char* POT_OUTPUT_FOLDER;
    int POINTS_X;
    int POINTS_Y;
    int POINTS_Z;
    double TIME_STEP;
    double STEP_X;
    double STEP_Y;
    double STEP_Z;
    double BETA;
    bool USE_IMAGINARY_TIME;
    bool USE_DYNAMIC_POTENTIAL;
    bool USE_FLOQUET_POTENTIAL;
    double FLOQUET_OMEGA;
    double FLOQUET_AMPLITUDE;

    double DP_CORRELATION_TIME;
    double DP_AMPLITUDE;
    int DP_PADDING_FACTOR_X;
    int DP_PADDING_FACTOR_Y;
    int DP_PADDING_FACTOR_Z;
    int DP_INTERPOLATION_STEPS;
    bool SAVE_POTENTIAL;
    bool SAVE_PSI = true;
    bool SAVE_REDUCED = false;

    double STOP_DELTA_ENERGY;
};

void read_config(const char* config_file, void* ptr);
//void smaller_into_bigger_array_1d(MKL_Complex16* input, MKL_Complex16* output, int size_smaller, int size_bigger);
//void write_file_to_output_folder(fftw_complex *input, int count);