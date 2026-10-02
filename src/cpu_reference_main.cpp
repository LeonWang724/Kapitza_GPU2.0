// Codex CUDA Port: 1D-only entry point around the untouched MKL CPU source.
#include <iostream>

#include "mkl.h"
#include <config.h>
#include <math_routines_1d.h>

int main(int argc, char* argv[]) {
    config_data configuration;
    if (argc == 1) {
        std::cout << "Using standard config file.\n";
        read_config("gpe1d.config", &configuration);
    } else if (argc == 2) {
        read_config(argv[1], &configuration);
    } else {
        std::cerr << "Expected zero arguments or one configuration path.\n";
        return 1;
    }

    if (configuration.dimension != 1) {
        std::cerr << "The validation CPU reference supports dimension=1 only.\n";
        return 1;
    }

    std::cout << "Threads available: " << mkl_get_max_threads() << '\n';
    if (configuration.NUM_THREADS != 0) {
        mkl_set_dynamic(false);
        mkl_set_num_threads(configuration.NUM_THREADS);
    }

    Math_routines_1d solver;
    solver.init_1d(&configuration);
    return solver.start_calculation_1d(&configuration);
}
