#include "config.h"
#include <iostream>
#include <fstream>
#include <sstream>   
#include <string>
#include <math.h>
#include <stdio.h>
#include <memory.h>



using namespace std;

/*
Read in the config file and fill config_data struct.
*/
void read_config(const char* config_file, void* ptr) {
    struct config_data* conf = (struct config_data*)ptr;
    ifstream myfile(config_file);
    string line;
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            istringstream is_line(line);
            string key;
            if (getline(is_line, key, '='))
            {
                string value;
                if (getline(is_line, value)) {
                    if (key.compare("number_of_threads") == 0) {
                        if (value.compare("dynamic") == 0)
                            conf->NUM_THREADS = 0;
                        else
                            conf->NUM_THREADS = stod(value);
                    }
                    else if (key.compare("number_of_iterations") == 0) {
                        conf->NUM_ITERATIONS = stod(value);
                    }
                    else if (key.compare("save_every_nth_iteration") == 0) {
                        conf->SAVE_STEP = stod(value);
                    }
                    else if (key.compare("show_stats_every_nth_iteration") == 0) {
                        conf->SHOW_STATS_STEP = stod(value);
                    }
                    else if (key.compare("potential_file") == 0) {
                        conf->POTENTIAL_FILE = strdup(value.c_str());
                    }
                    else if (key.compare("floquet_potential_file") == 0) {
                        conf->FLOQUET_POTENTIAL_FILE = strdup(value.c_str());
                    }
                    else if (key.compare("status_file") == 0) {
                        conf->STATUS_FILE = strdup(value.c_str());
                    }
                    else if (key.compare("initial_state_file") == 0) {
                        conf->INITIAL_FILE = strdup(value.c_str());
                    }
                    else if (key.compare("output_folder") == 0) {
                        conf->OUTPUT_FOLDER = strdup(value.c_str());
                    }
                    else if (key.compare("time_step") == 0) {
                        conf->TIME_STEP = stod(value);
                    }
                    else if (key.compare("stop_itime_on_energy_difference") == 0) {
                        conf->STOP_DELTA_ENERGY = stod(value);
                    }
                    else if (key.compare("points_x") == 0) {
                        conf->POINTS_X = stod(value);
                    }
                    else if (key.compare("points_y") == 0) {
                        conf->POINTS_Y = stod(value);
                    }
                    else if (key.compare("points_z") == 0) {
                        conf->POINTS_Z = stod(value);
                    }
                    else if (key.compare("step_x") == 0) {
                        conf->STEP_X = stod(value);
                    }
                    else if (key.compare("step_y") == 0) {
                        conf->STEP_Y = stod(value);
                    }
                    else if (key.compare("step_z") == 0) {
                        conf->STEP_Z = stod(value);
                    }
                    else if (key.compare("beta") == 0) {
                        conf->BETA = stod(value);
                    }
                    else if (key.compare("dimension") == 0) {
                        conf->dimension = stod(value);
                    }
                    else if (key.compare("imaginary_time") == 0) {
                        if (value.compare("true") == 0) {
                            conf->USE_IMAGINARY_TIME = true;
                        }
                        else {
                            conf->USE_IMAGINARY_TIME = false;
                        }
                    }
                    else if (key.compare("floquet_potential") == 0) {
                        if (value.compare("true") == 0) {
                            conf->USE_FLOQUET_POTENTIAL = true;
                        }
                        else {
                            conf->USE_FLOQUET_POTENTIAL = false;
                        }
                    }
                    else if (key.compare("floquet_omega") == 0) {
                        conf->FLOQUET_OMEGA = stod(value);
                    }
                    else if (key.compare("floquet_amplitude") == 0) {
                        conf->FLOQUET_AMPLITUDE = stod(value);
                    }
                    else if (key.compare("dynamic_potential") == 0) {
                        if (value.compare("true") == 0) {
                            conf->USE_DYNAMIC_POTENTIAL = true;
                        }
                        else {
                            conf->USE_DYNAMIC_POTENTIAL = false;
                        }
                    }
                    else if (key.compare("dp_correlation_time") == 0) {
                        conf->DP_CORRELATION_TIME = stod(value);
                    }
                    else if (key.compare("dp_amplitude") == 0) {
                        conf->DP_AMPLITUDE = stod(value);
                    }
                    else if (key.compare("dp_padding_factor_x") == 0) {
                        conf->DP_PADDING_FACTOR_X = stod(value);
                    }
                    else if (key.compare("dp_padding_factor_y") == 0) {
                        conf->DP_PADDING_FACTOR_Y = stod(value);
                    }
                    else if (key.compare("dp_padding_factor_z") == 0) {
                        conf->DP_PADDING_FACTOR_Z = stod(value);
                    }
                    else if (key.compare("dp_interpolation_steps") == 0) {
                        conf->DP_INTERPOLATION_STEPS = stod(value);
                    }
                    else if (key.compare("dp_save_potential") == 0) {
                        if (value.compare("true") == 0) {
                            conf->SAVE_POTENTIAL = true;
                        }
                        else {
                            conf->SAVE_POTENTIAL = false;
                        }
                    }
                    else if (key.compare("dp_potential_output_folder") == 0) {
                        conf->POT_OUTPUT_FOLDER = strdup(value.c_str());
                    }
                    else if (key.compare("save_psi") == 0) {
                        if (value.compare("true") == 0) {
                            conf->SAVE_PSI = true;
                        }
                        else {
                            conf->SAVE_PSI = false;
                        }
                    }
                    else if (key.compare("save_reduced") == 0) {
                        if (value.compare("true") == 0) {
                            conf->SAVE_REDUCED = true;
                        }
                        else {
                            conf->SAVE_REDUCED = false;
                        }
                    }
                }
            }
        }
    }
    myfile.close();
}