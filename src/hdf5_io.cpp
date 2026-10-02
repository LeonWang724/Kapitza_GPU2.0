// Codex CUDA Port: exact REAL/IMAGINARY double-precision HDF5 layout and CSV output.
#include "hdf5_io.h"

#include <hdf5.h>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

class Hdf5Handle {
public:
    using Closer = herr_t (*)(hid_t);
    Hdf5Handle(hid_t value, Closer closer) : value_(value), closer_(closer) {}
    ~Hdf5Handle() {
        if (value_ >= 0) {
            closer_(value_);
        }
    }
    Hdf5Handle(const Hdf5Handle&) = delete;
    Hdf5Handle& operator=(const Hdf5Handle&) = delete;
    hid_t get() const noexcept { return value_; }

private:
    hid_t value_;
    Closer closer_;
};

void require_hdf5(herr_t status, const std::string& action) {
    if (status < 0) {
        throw std::runtime_error("HDF5 failure while " + action);
    }
}

Hdf5Handle open_dataset(hid_t file, const char* name) {
    const hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dataset < 0) {
        throw std::runtime_error(std::string("Missing HDF5 dataset: ") + name);
    }
    return Hdf5Handle(dataset, H5Dclose);
}

}  // namespace

std::vector<cuDoubleComplex> read_complex_hdf5(
    const std::filesystem::path& path, std::size_t expected_size) {
    const hid_t file_id = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Unable to open HDF5 input: " + path.string());
    }
    Hdf5Handle file(file_id, H5Fclose);
    auto real_dataset = open_dataset(file.get(), "REAL");
    auto imag_dataset = open_dataset(file.get(), "IMAGINARY");

    Hdf5Handle space(H5Dget_space(real_dataset.get()), H5Sclose);
    if (space.get() < 0 || H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error("Expected a one-dimensional REAL dataset in " + path.string());
    }
    hsize_t dimensions[1] = {0};
    H5Sget_simple_extent_dims(space.get(), dimensions, nullptr);
    if (dimensions[0] != expected_size) {
        throw std::runtime_error("HDF5 length does not match points_x in " + path.string());
    }

    std::vector<double> real(expected_size);
    std::vector<double> imag(expected_size);
    require_hdf5(H5Dread(real_dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, real.data()),
                 "reading REAL from " + path.string());
    require_hdf5(H5Dread(imag_dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, imag.data()),
                 "reading IMAGINARY from " + path.string());

    std::vector<cuDoubleComplex> values(expected_size);
    for (std::size_t index = 0; index < expected_size; ++index) {
        values[index] = make_cuDoubleComplex(real[index], imag[index]);
    }
    return values;
}

void write_complex_hdf5(const std::filesystem::path& path,
                        const std::vector<cuDoubleComplex>& values) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::vector<double> real(values.size());
    std::vector<double> imag(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        real[index] = cuCreal(values[index]);
        imag[index] = cuCimag(values[index]);
    }

    const hid_t file_id = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Unable to create HDF5 output: " + path.string());
    }
    Hdf5Handle file(file_id, H5Fclose);
    const hsize_t dimensions[1] = {static_cast<hsize_t>(values.size())};
    Hdf5Handle space(H5Screate_simple(1, dimensions, nullptr), H5Sclose);
    if (space.get() < 0) {
        throw std::runtime_error("Unable to create HDF5 dataspace for " + path.string());
    }

    const hid_t real_id = H5Dcreate2(file.get(), "REAL", H5T_IEEE_F64LE, space.get(),
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const hid_t imag_id = H5Dcreate2(file.get(), "IMAGINARY", H5T_IEEE_F64LE, space.get(),
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (real_id < 0 || imag_id < 0) {
        if (real_id >= 0) H5Dclose(real_id);
        if (imag_id >= 0) H5Dclose(imag_id);
        throw std::runtime_error("Unable to create REAL/IMAGINARY datasets in " +
                                 path.string());
    }
    Hdf5Handle real_dataset(real_id, H5Dclose);
    Hdf5Handle imag_dataset(imag_id, H5Dclose);
    require_hdf5(H5Dwrite(real_dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, real.data()),
                 "writing REAL to " + path.string());
    require_hdf5(H5Dwrite(imag_dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, imag.data()),
                 "writing IMAGINARY to " + path.string());
}

std::filesystem::path snapshot_path(const std::filesystem::path& output_folder,
                                    int iteration) {
    std::ostringstream name;
    name << std::setw(16) << std::setfill('0') << iteration << ".h5";
    return output_folder / name.str();
}

StatusWriter::StatusWriter(const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    stream_.open(path, std::ios::trunc);
    if (!stream_) {
        throw std::runtime_error("Unable to create status CSV: " + path.string());
    }
    // CUDA PORT: matches the workflow executable, which adds InitOverlap to the
    // eight-column source CSV. The diagnostic definition is documented in notes.
    stream_ << "Iteration;Norm;Etot;Ekin;Epot;Eint;Chpot;Meandynpot;InitOverlap\n";
}

void StatusWriter::append(const StatusData& status) {
    stream_ << std::setprecision(12) << status.iteration << ';' << status.norm << ';'
            << status.total_energy << ';' << status.kinetic_energy << ';'
            << status.potential_energy << ';' << status.interaction_energy << ';'
            << status.chemical_potential << ';' << status.mean_dynamic_potential << ';'
            << status.initial_density_overlap << '\n';
    stream_.flush();
    if (!stream_) {
        throw std::runtime_error("Failed while writing the status CSV.");
    }
}

