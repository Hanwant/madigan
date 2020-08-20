#include "synth.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;



PYBIND11_MODULE(synth_c, m){
  m.doc() = "Synths written in C++";
  py::class_<SineGen>(m, "SineGen")
    .def(py::init<>())
    .def(py::init<double, double, double>(), py::arg("mu"), py::arg("amp"), py::arg("dx"))
    .def("__next__", &SineGen::render)
    .def("render", &SineGen::render);

}
