#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
#include "Portfolio.h"
#include "DataSource.h"
namespace py = pybind11;

using namespace madigan;

PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  // m.def("movingWindowMean", py::overload_cast<Matrix<time_t, Dynamic, 1>, Matrix<double, Dynamic, 1>, int>(&movingWindowMean), "discrete window moving mean",
  //       py::arg("timestamps"), py::arg("price"), py::arg("window"), py::return_value_policy::copy);


  py::class_<Portfolio>(m, "Portfolio")
    .def(py::init<int, double> (), py::arg("nAssets"), py::arg("initCash")=double(1'000'000))
    .def("portfolio", (vector<double>(Portfolio::*)()) &Portfolio::portfolio,
         "Return vector of asset holdings",
         py::return_value_policy::copy);

  py::class_<DataSource>(m, "DataSource");
  py::class_<Synth, DataSource>(m, "Synth")
    .def(py::init<>())
    .def(py::init<
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"))
    .def("getData", (vector<double> (Synth::*) ()) &Synth::getData,
                     "Get Next data points",
                     py::return_value_policy::copy);
}
