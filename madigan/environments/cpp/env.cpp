#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "DataSource.h"
#include "Portfolio.h"
#include "Account.h"
#include "Broker.h"
namespace py = pybind11;

using namespace madigan;

// PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(PriceVector);

PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  py::class_<Asset>(m, "Asset")
    .def(py::init<string> (), py::arg("asset_name"))
    .def(py::init<string, string> (), py::arg("asset_name"), py::arg("exchange"))
    .def_readwrite("name", &Asset::name)
    .def_readwrite("code", &Asset::code)
    .def_readwrite("exchange", &Asset::exchange)
    .def_readwrite("bp_multiplier", &Asset::bpMultiplier)
    .def("__repr__", [] (const Asset& a){
      return "name: " + a.name + " code: " + a.code;
    });

  // py::bind_vector<Assets>(m, "Assets");
  py::class_<Assets>(m, "Assets")
    .def(py::init<std::vector<string>> (), py::arg("asset_names_list"))
    .def(py::init<std::vector<Asset>> (), py::arg("asset_names_list"))
    .def("__len__", [](const Assets &v) { return v.size(); })
    .def("__repr__", [](const Assets &v) {
      std::stringstream repr;
      repr << "[";
      if(v.size() > 0){
        for (auto asset: v){
          repr << asset.name << ", ";
        }
      }
      repr << "]\n";
      return repr.str();})
    .def("__iter__", [](Assets &v) {
      return py::make_iterator(v.begin(), v.end());
    }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */
    .def("__getitem__", [](Assets &v, int idx) {
      return v[idx];
    });



  py::class_<PriceVector>(m, "PriceVector", py::buffer_protocol())
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
    .def("getData", (PriceVector& (Synth::*) ()) &Synth::getData,
         "Get Next data points",
         py::return_value_policy::reference);
    // .def("getData", (py::array_t<double> (Synth::*) ()) &Synth::getData_np,
    //      "Get Next data points",
    //      py::return_value_policy::reference);

  py::class_<Account>(m, "Account")
    .def(py::init<>())
    .def(py::init<Portfolio> (), py::arg("portfolio"))
    .def(py::init<string, Assets, double> (), py::arg("id"), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def(py::init<Assets, double> (), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def("portfolio", (Portfolio(Account::*)()) &Account::portfolio,
         "Return default porfolio",
         py::return_value_policy::copy)
    .def("portfolios", (PortfolioBook(Account::*)()) &Account::portfolios,
         "Return dict of porfolios",
         py::return_value_policy::copy);

  py::class_<Portfolio>(m, "Portfolio")
    .def(py::init<>())
    .def(py::init<string, Assets, double> (), py::arg("id"), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def("portfolio", (Ledger(Portfolio::*)()) &Portfolio::portfolio,
         "Return vector of asset holdings",
         py::return_value_policy::copy);

  py::class_<Broker>(m, "Broker")
    .def(py::init<>())
    .def(py::init<Account> (), py::arg("account"))
    .def(py::init<Portfolio> (), py::arg("portfolio"))
    .def(py::init<string, Assets, double> (), py::arg("AccId"), py::arg("assets"), py::arg("initCash")=double(1'000'000));

}
