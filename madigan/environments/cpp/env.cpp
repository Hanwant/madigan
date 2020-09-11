#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
#include "Portfolio.h"
#include "Env.h"
namespace py = pybind11;


PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  // m.def("movingWindowMean", py::overload_cast<Matrix<time_t, Dynamic, 1>, Matrix<double, Dynamic, 1>, int>(&movingWindowMean), "discrete window moving mean",
  //       py::arg("timestamps"), py::arg("price"), py::arg("window"), py::return_value_policy::copy);


  py::class_<Portfolio>(m, "Portfolio")
    .def(py::init<int, double> (), py::arg("nAssets"), py::arg("initCash")=double(1'000'000))
    .def_readonly("portfolio", &RollerX::_portfolio)
    .def("portfolio", (vector<double>(Portfolio::*)()) &Portfolio::portfolio,
         "Return vector of asset holdings";
         py::return_value_policy::copy)
}
