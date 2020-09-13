#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <pybind11/eigen.h>

#include "Portfolio.h"
#include "Account.h"
#include "DataSource.h"
namespace py = pybind11;

using namespace madigan;

// PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  // m.def("movingWindowMean", py::overload_cast<Matrix<time_t, Dynamic, 1>, Matrix<double, Dynamic, 1>, int>(&movingWindowMean), "discrete window moving mean",
  //       py::arg("timestamps"), py::arg("price"), py::arg("window"), py::return_value_policy::copy);

  py::class_<PriceVector>(m, "PriceVector", py::buffer_protocol())
    // .def(py::init<>())
    // .def("clear", &std::vector<double>::clear)
    // // .def("pop_back", &std::vector<double>::pop_back)
    // .def("__len__", [](const std::vector<double> &v) { return v.size(); })
    // .def("__repr__", [](const std::vector<double> &v) {
    //   std::stringstream repr;
    //   if(v.size() >1){
    //     repr << "["<< v[0] << " ... " << *(v.end()-1 )<< "]\n";
    //   }
    //   else if(v.size() == 1){
    //     repr << "["<< v[0] << "]\n";
    //   }
    //   else{
    //     repr << "[]\n";
    //   }
    //   return repr.str();})
    // .def("__iter__", [](std::vector<double> &v) {
    //   return py::make_iterator(v.begin(), v.end());
    // }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */
    .def_buffer([](PriceVector &v)->py::buffer_info {
      return py::buffer_info(
                             v.data(),
                             sizeof(double),
                             py::format_descriptor<double>::format(),
                             1, // ndim
                             {v.size()}, // dim
                             {sizeof(double)} // strides
                             );
    });

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
    .def("getData", (const PriceVector& (Synth::*) ()) &Synth::getData,
         "Get Next data points",
         py::return_value_policy::reference);
    // .def("getData", (py::array_t<double> (Synth::*) ()) &Synth::getData_np,
    //      "Get Next data points",
    //      py::return_value_policy::reference);

  py::class_<Portfolio>(m, "Portfolio")
    .def(py::init<string, int, double> (), py::arg("id"), py::arg("nAssets"), py::arg("initCash")=double(1'000'000))
    .def("portfolio", (vector<double>(Portfolio::*)()) &Portfolio::portfolio,
         "Return vector of asset holdings",
         py::return_value_policy::copy);

  py::class_<Account>(m, "Account")
    .def(py::init<string, int, double> (), py::arg("id"), py::arg("nAssets"), py::arg("initCash")=double(1'000'000));
  // .def(py::init<int, double> (), py::arg("nAssets"), py::arg("initCash")=double(1'000'000));
    // .def("portfolios", (vector<double>(Account::*)()) &Account::portfolio,
    //      "Return vector of asset holdings",
    //      py::return_value_policy::copy);
}
