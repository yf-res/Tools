// python_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "fast_nuc.hpp"

namespace py = pybind11;

class PyFastNUC {
public:
    PyFastNUC(py::array_t<uint16_t> gain, py::array_t<uint16_t> offset) {
        if (gain.ndim() != 2 || offset.ndim() != 2)
            throw std::runtime_error("Input must be 2-D NumPy array");
        if (gain.shape(0) != offset.shape(0) || gain.shape(1) != offset.shape(1))
            throw std::runtime_error("Gain and offset must have the same shape");

        height = gain.shape(0);
        width = gain.shape(1);

        auto gain_buf = gain.request();
        auto offset_buf = offset.request();
        uint16_t* gain_ptr = static_cast<uint16_t*>(gain_buf.ptr);
        uint16_t* offset_ptr = static_cast<uint16_t*>(offset_buf.ptr);

        std::vector<uint16_t> gain_vec(gain_ptr, gain_ptr + height * width);
        std::vector<uint16_t> offset_vec(offset_ptr, offset_ptr + height * width);

        nuc = std::make_unique<FastNUC>(gain_vec, offset_vec, width, height);
    }

    py::array_t<uint16_t> correct(py::array_t<uint16_t> input) {
        if (input.ndim() != 2)
            throw std::runtime_error("Input must be 2-D NumPy array");
        if (input.shape(0) != height || input.shape(1) != width)
            throw std::runtime_error("Input shape must match gain and offset shape");

        auto input_buf = input.request();
        uint16_t* input_ptr = static_cast<uint16_t*>(input_buf.ptr);

        std::vector<uint16_t> input_vec(input_ptr, input_ptr + height * width);
        std::vector<uint16_t> output_vec = nuc->correct(input_vec);

        auto result = py::array_t<uint16_t>({height, width});
        auto result_buf = result.request();
        uint16_t* result_ptr = static_cast<uint16_t*>(result_buf.ptr);
        std::copy(output_vec.begin(), output_vec.end(), result_ptr);

        return result;
    }

private:
    std::unique_ptr<FastNUC> nuc;
    int width, height;
};

PYBIND11_MODULE(fast_nuc, m) {
    py::class_<PyFastNUC>(m, "FastNUC")
        .def(py::init<py::array_t<uint16_t>, py::array_t<uint16_t>>())
        .def("correct", &PyFastNUC::correct);
}