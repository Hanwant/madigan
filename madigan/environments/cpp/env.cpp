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
#include "Env.h"
#include "Config.h"

namespace py = pybind11;
using namespace madigan;

// PYBIND11_MAKE_OPAQUE(PortfolioBook);
// PYBIND11_MAKE_OPAQUE(std::unordered_map<string, Portfolio>);
// PYBIND11_MAKE_OPAQUE(vector<Portfolio>);


PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  // py::class_<Config>(m, "ConfigC")
    // .def(py::init<py::dict> (), py::arg("config"))
    // .def(py::init<Config> (), py::arg("config"));
    // .def(py::init<py::dict> (), py::arg("dict"));

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
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"))
    .def("getData", (PriceVector& (Synth::*) ()) &Synth::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (Synth::*) ()) &Synth::currentData,
         "Get current data points",
         py::return_value_policy::reference);

  py::class_<Portfolio>(m, "Portfolio")
    // .def(py::init<>())
    .def(py::init<string, Assets, double> (),
         py::arg("id"),
         py::arg("assets"),
         py::arg("initCash")=double(1'000'000))
    .def("setDataSource", &Portfolio::setDataSource,
         "assign data source for current prices reference")
    .def_property_readonly("id", &Portfolio::id, "portfolio id")
    .def_property_readonly("equity", &Portfolio::equity,
                           "returns net equity")
    .def_property_readonly("cash", &Portfolio::cash,
                           "returns net cash")
    .def_property_readonly("borrowedMargin", &Portfolio::borrowedMargin,
                           "Margin currently borrowed (i.e for levaraged buy)")
    .def_property_readonly("currentPrices", &Portfolio::currentPrices,
                           "current prices as per registered data source", py::return_value_policy::copy)
    .def_property_readonly("nAssets", &Portfolio::nAssets,
                           "number of assets")
    .def_property_readonly("assets", (Assets(Portfolio::*)()) &Portfolio::assets,
                           "Returns list of Asset objects")
    .def_property_readonly("ledger", &Portfolio::ledger,
                           "vector of asset holdings",
                           py::return_value_policy::reference) /* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                  AND CONNECTED TO DATA SOURCE*/
    .def("handleTransaction", (void(Portfolio::*)(int, double, double, double,
                                                  double )) &Portfolio::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("assetIdx"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0., py::arg("requiredmargin")=1.0)
    .def("handleTransaction", (void(Portfolio::*)(string, double, double, double,
                                                  double )) &Portfolio::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("asset"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0., py::arg("requiredmargin")=1.0)
    .def("__repr__", [](const Portfolio &port) {
      std::stringstream repr;
      repr << port.id() << ": {";
      repr << " cash: " << port.cash();
      repr << " equity: " << port.equity();
      repr << " borrowedMargin: " << port.borrowedMargin();
      repr << "}" << "\n";
      return repr.str();});

  // py::bind_map<PortfolioBook>(m, "PortfolioBook");
    // .def("__repr__", [](const PortfolioBook &ports) {
    //   std::stringstream repr;
    //   repr << "[";
    //   if(ports.size() > 0){
    //     for (auto port: ports){
    //       repr << port.first << ": " << *(port.second);
    //       std::cout<<port.first << *port.second << "\n";
    //     }
    //   }
    //   repr << "]\n";
    //   return repr.str();});
  // py::bind_map<PortfolioBook>(m, "PortfolioBook");
  py::bind_map<std::unordered_map<string, Portfolio>>(m, "PortfolioBook");

  py::class_<Account>(m, "Account")
    // .def(py::init<>())
    .def(py::init<Portfolio&> (), py::arg("portfolio"))
    .def(py::init<string, Assets, double> (), py::arg("id"),
         py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def(py::init<Assets, double> (), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def_property_readonly("id", &Account::id, "account id")
    .def("addPortfolio", (void (Account::*) (py::object)) &Account::addPortfolio,
         "add portfolio to account. Must contain the same number of assets as the others",
         py::arg("portfolio"))
    .def("addPortfolio", (void (Account::*) (string, Assets, double)) &Account::addPortfolio,
         "add portfolio to account. Must contain the same number of assets as the others",
         py::arg("portID"), py::arg("Assets"), py::arg("initCash"))
    .def("addPortfolio", (void (Account::*) (Assets, double)) &Account::addPortfolio,
         "add portfolio to account. Must contain the same number of assets as the others",
         py::arg("Assets"), py::arg("initCash"))
    .def("setDataSource", &Account::setDataSource,
         "assign data source for current prices reference")
    .def("equity", (double(Account::*)() const) &Account::equity,
         "returns net equity")
    .def("cash", (double(Account::*)() const) &Account::cash,
         "returns net cash")
    .def("borrowedMargin", (double(Account::*)() const) &Account::borrowedMargin,
         "returns net borrowedMargin")
    .def("currentPrices",  &Account::currentPrices,
         "returns net currentPrices", py::return_value_policy::reference)
    .def("nAssets", &Account::nAssets,
         "number of assets")
    .def("assets", (Assets(Account::*)()) &Account::assets,
         "Returns list of Asset objects")
    .def("equity", (double(Account::*)(string) const) &Account::equity,
         "returns net equity")
    .def("cash", (double(Account::*)(string) const) &Account::cash,
         "returns net cash")
    .def("borrowedMargin", (double(Account::*)(string) const) &Account::borrowedMargin,
         "returns net borrowedMargin")
    .def("handleTransaction", (void(Account::*)(int, double, double, double,
                                                  double )) &Account::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("assetIdx"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0., py::arg("requiredmargin")=1.0)
    .def("handleTransaction", (void(Account::*)(string, double, double, double,
                                                  double )) &Account::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("asset"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0., py::arg("requiredmargin")=1.0)
    .def("handleTransaction", (void(Account::*)(string, int, double, double, double,
                                                double )) &Account::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("portID"), py::arg("assetIdx"), py::arg("transactionPrice"),
         py::arg("units"), py::arg("transactionCost")=0.,
         py::arg("requiredmargin")=1.0)
    .def("handleTransaction", (void(Account::*)(string, string, double, double, double,
                                                double )) &Account::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("pordID"), py::arg("asset"), py::arg("transactionPrice"),
         py::arg("units"), py::arg("transactionCost")=0.,
         py::arg("requiredmargin")=1.0)
    .def("portfolio", (const Portfolio&(Account::*)() const) &Account::portfolio,
         "Return default porfolio",
         py::return_value_policy::copy)
    .def("portfolio", (const Portfolio&(Account::*)(string) const) &Account::portfolio,
         "Return default porfolio",
         py::return_value_policy::copy)
    // .def("portfolios", (std::vector<Portfolio>&(Account::*)()) &Account::portfolios,
    //      "Return list of porfolios",
    //      py::return_value_policy::copy)
    .def("portfolios", &Account::portfolios,
         "Return list of porfolios",
         py::return_value_policy::copy)
    .def("portfolioBook", &Account::portfolioBookCopy,
         "Return dict of porfolios",
         py::return_value_policy::copy)
    // .def("portfolioBookp", &Account::py_portfolioBook,
    //      "Return dict of porfolios",
    //      py::return_value_policy::reference)
    .def("ledger", (Ledger (Account::*)() const) &Account::ledger,
         "Return vector of current asset holdings",
         py::return_value_policy::copy)
    .def("ledger", (const Ledger& (Account::*)(string) const) &Account::ledger,
         "Return vector of current asset holdings",
         py::return_value_policy::copy);

  py::class_<Broker>(m, "Broker")
    // .def(py::init<>())
    // .def(py::init<Account&> (), py::arg("account"))
    // .def(py::init<Portfolio&> (), py::arg("portfolio"))
    .def(py::init<Assets, double> (), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def(py::init<string, Assets, double> (),
         py::arg("AccId"), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def("account", (const Account& (Broker::*)() const) &Broker::account,
         "Returns default account - useful if just one acc",
         py::return_value_policy::reference)
    .def("account", (const Account& (Broker::*)(string) const) &Broker::account,
         "Returns acc by accID. Throws out_of_range error if accID doesn't exist",
         py::return_value_policy::reference)
    .def("accounts", (std::vector<Account>&(Broker::*)()) &Broker::accounts,
         "Return list of accounts",
         py::return_value_policy::reference);

  py::class_<Env>(m, "Env")
    .def(py::init<string, Assets, double> (),
         py::arg("dataSourceType"),
         py::arg("assets"),
         py::arg("initCash")=double(1'000'000))
    .def(py::init<string, Assets, double, py::dict> (),
         py::arg("dataSourceType"),
         py::arg("assets"),
         py::arg("initCash"),
         py::arg("config_dict"));

}

//======================================================================
//==================  Testing Passing data to and from python ==========
//======================================================================
// typedef std::unordered_map<string, int> strintMap;
// typedef std::unordered_map<string, std::any> stranyMap;
// template<typename T>
// struct takesDict{
//   T map;
//   takesDict(T map);
//   takesDict(py::dict map);
//   // T dict() { return map;}
//   py::dict dict() { return map;}
// };
// template<>
// takesDict<stranyMap>::takesDict(stranyMap map): map(map){
// }
// template<>
// takesDict<strintMap>::takesDict(strintMap map): map(map){
// }
// template<>
// takesDict<strintMap>::takesDict(py::dict map){
//   strintMap map_;
//   std::cout<< "INITING FROM PYDICT\n";
//   for(auto item: map){
//     map_[std::string(py::str(item.first))] = item.second.cast<int>();
//   }
//   this->map = map_;
// }

// template<>
// takesDict<stranyMap>::takesDict(py::dict pydict){
//   stranyMap map_;
//   std::cout<< "INITING FROM PYDICT STRANY\n";
//   auto found = std::find_if(pydict.begin(), pydict.end(), [](const std::pair<py::handle, py::handle>& pair){
//     return string(py::str(pair.first)) == "c";
//   });
//   if (found != pydict.end()){
//     std::cout << " c FOUND\n";
//   } else  std::cout << " c NOT FOUND\n";
//   for(auto item: pydict){
//     string key = std::string(py::str(item.first));
//     std::cout << "value type: " << item.second.get_type() << "\n";
//     std::cout << "value type: " << py::isinstance<py::int_>(item.second) << "\n";
//     if (key == "a"){
//       if (py::isinstance<py::int_>(item.second)){
//         map_[key] = item.second.cast<int>();
//       }
//     }
//     if (key == "b"){
//       if (py::isinstance<py::int_>(item.second)){
//         map_[key] = item.second.cast<int>();
//       }
//     }
//   }
//   this->map = map_;
// }
// template<>
// py::dict takesDict<stranyMap>::dict(){
//   py::dict out;
//   std::cout << "starting\n";
//   for (auto item: map){
//     string key = item.first;
//     std::cout << "item : " << key << "\n";
//     if (key=="a"){
//       out[py::str(key)] = std::any_cast<int>(item.second);
//     }
//     if (key=="b"){
//       out[py::str(key)] = std::any_cast<int>(item.second);
//     }
//   }
//   std::cout << "end\n";
//   return out;
// }
// PYBIND11_MODULE(env, m){
//   m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

//   // py::class_<Config>(m, "ConfigC")
//   // .def(py::init<py::dict> (), py::arg("config"))
//   // .def(py::init<Config> (), py::arg("config"));
//   // .def(py::init<py::dict> (), py::arg("dict"));

//   typedef stranyMap mapType;
//   py::class_<takesDict<mapType>>(m, "takesDict")
//     .def(py::init<py::dict> (), py::arg("pydict"))
//     .def(py::init<mapType> (), py::arg("map"))
//     // .def("dict", (mapType(takesDict<mapType>::*)()) &takesDict<mapType>::dict);
//     .def("dict", (py::dict(takesDict<mapType>::*)()) &takesDict<mapType>::dict);
// }
