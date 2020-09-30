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
// PYBIND11_MAKE_OPAQUE(std::tuple);
PYBIND11_MAKE_OPAQUE(SRDI<double>);
PYBIND11_MAKE_OPAQUE(SRDI<PriceVector>);
// using SRDI = std::tuple<State, double, bool, EnvInfo<T>>;

template<typename T>
void declareBrokerResponse(py::module& m, const string& className){
  using Class = BrokerResponse<T>;
  py::class_<Class>(m, className.c_str())
    .def_readonly("event", &Class::event)
    .def_readonly("timestamp", &Class::timestamp)
    .def_readonly("transactionPrice", &Class::transactionPrice)
    .def_readonly("transactionCost", &Class::transactionCost)
    .def_readonly("riskInfo", &Class::riskInfo)
    .def("__repr__", [](const BrokerResponse<T>& b){
        std::stringstream repr;
        repr << "timestamp:        " << b.timestamp << "\n";
        repr << "transactionPrice: \n" << b.transactionPrice << "\n";
        repr << "transactionCost:  \n" << b.transactionCost << "\n";
        repr << "riskInfo:         \n" << b.riskInfo << "\n";
        return repr.str();}
      );
}

template<typename T>
void declareEnvInfo(py::module& m, const string& className){
  using Class = EnvInfo<T>;
  py::class_<Class>(m, className.c_str())
    .def(py::init<> (), "default empty constructor")
    .def(py::init<BrokerResponse<T>> (), py::arg("brokerResponse"))
    .def_readonly("brokerResponse", &Class::brokerResponse);
}

PYBIND11_MODULE(env, m){
  m.doc() = "Environment Components, including Portfolio, Broker, DataGenerator and Env classes";

  // py::class_<Config>(m, "ConfigC")
    // .def(py::init<py::dict> (), py::arg("config"))
    // .def(py::init<Config> (), py::arg("config"));
    // .def(py::init<py::dict> (), py::arg("dict"));

  // py::bind_vector<Assets>(m, "Assets");

  py::enum_<RiskInfo> _RiskInfo(m, "RiskInfo");
  _RiskInfo.value("green", RiskInfo::green)
    .value("margin_call", RiskInfo::margin_call)
    .value("insuff_margin", RiskInfo::insuff_margin)
    .value("blown_out", RiskInfo::blown_out);

  declareEnvInfo<double>(m, "EnvInfoSingle");
  declareEnvInfo<PriceVector>(m, "EnvInfoMulti");
  declareBrokerResponse<double>(m, "BrokerResponseSingle");
  declareBrokerResponse<PriceVector>(m, "BrokerResponseMulti");

  // Declared and defined in the same order
  py::class_<Asset> _Asset(m, "Asset");
  py::class_<Assets> _Assets(m, "Assets");
  py::class_<PriceVector> _PriceVector(m, "PriceVector", py::buffer_protocol());
  py::bind_map<std::unordered_map<string, Portfolio>>(m, "PortfolioBook");
  py::bind_map<std::unordered_map<string, Account>>(m, "AccountBook");

  py::class_<DataSource>_DataSource(m, "DataSource");
  py::class_<Synth, DataSource>_Synth(m, "Synth");

  py::class_<Portfolio>_Portfolio(m, "Portfolio");
  py::class_<Account>_Account(m, "Account");
  py::class_<Broker> _Broker(m, "Broker");
  py::class_<Env> _Env(m, "Env");


  _Asset.def(py::init<string> (), py::arg("asset_name"))
    .def(py::init<string, string> (), py::arg("asset_name"), py::arg("exchange"))
    .def_readwrite("name", &Asset::name)
    .def_readwrite("code", &Asset::code)
    .def_readwrite("exchange", &Asset::exchange)
    .def_readwrite("bp_multiplier", &Asset::bpMultiplier)
    .def("__repr__", [] (const Asset& a){
      return "name: " + a.name + " code: " + a.code;
    });

  _Assets.def(py::init<std::vector<string>> (), py::arg("asset_names_list"))
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

  _PriceVector.def_buffer([](PriceVector &v)->py::buffer_info {
      return py::buffer_info(
                             v.data(),
                             sizeof(double),
                             py::format_descriptor<double>::format(),
                             1, // ndim
                             {v.size()}, // dim
                             {sizeof(double)} // strides
                             );
    });

  _Synth.def(py::init<>())
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

    // .def(py::init<>())
  _Portfolio.def(py::init<string, Assets, double> (),
         py::arg("id"),
         py::arg("assets"),
         py::arg("initCash")=double(1'000'000))
    .def("setDataSource", &Portfolio::setDataSource,
         "assign data source for current prices reference")
    .def_property_readonly("id", &Portfolio::id, "portfolio id")
    .def_property_readonly("nAssets", &Portfolio::nAssets,
                           "number of assets")
    .def_property_readonly("assets", (Assets(Portfolio::*)()) &Portfolio::assets,
                           "Returns list of Asset objects")
    .def_property_readonly("cash", &Portfolio::cash,
                           "returns current cash holdings")
    .def_property_readonly("balance", &Portfolio::balance,
                           "returns net cash balance")
    .def_property_readonly("equity", &Portfolio::equity,
                           "returns net equity")
    .def_property_readonly("pnl", &Portfolio::pnl,
                           "returns current net profit/loss")
    .def_property_readonly("borrowedMargin", &Portfolio::borrowedMargin,
                           "Total Margin currently borrowed for positions (i.e for levaraged buy)")
    .def_property_readonly("borrowedMarginLedger", &Portfolio::borrowedMarginLedger,
                           "Margin currently borrowed for each position (i.e for levaraged buy)")
    .def_property_readonly("availableMargin", &Portfolio::availableMargin,
                           "Margin available for entering positions (i.e for levaraged buy)")
    .def_property_readonly("assetValue", &Portfolio::assetValue,
                           "net value of current positions (long + short)")
    .def_property_readonly("borrowedAssetValue", &Portfolio::borrowedAssetValue,
                           "value of current short positions")
    .def_property_readonly("usedMargin", &Portfolio::usedMargin,
                           "Margin currently committed to long positions ")
    .def_property_readonly("currentPrices", &Portfolio::currentPrices,
                           "current prices as per registered data source "
                           "\n Careful as the returned array will contain a reference"
                           "to the datasource buffer - const is ignored",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND CONNECTED TO DATA SOURCE*/
    .def_property_readonly("ledger", &Portfolio::ledger,
                           "vector of asset holdings",
                           py::return_value_policy::reference)
    .def("setRequiredMargin", (void (Portfolio::*)(double)) &Portfolio::setRequiredMargin,
         "set required Margin level for port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("requiredMargin"))
    .def("setMaintenanceMargin", (void (Portfolio::*)(double)) &Portfolio::setMaintenanceMargin,
         "set maintenance Margin level for port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("maintenanceMargin"))
    .def("checkRisk", (RiskInfo (Portfolio::*) () const) &Portfolio::checkRisk,
         "checks current risk levels and returns a RiskInfo enum object\n"
         "I.e returns RiskInfo::green unless current equity is lower than "
         "maintenance margin requirements ")
    .def("checkRisk", (RiskInfo (Portfolio::*) (double) const) &Portfolio::checkRisk,
         "checks availableMargin and compares with proposed amount_to_purchase"
         "I.e returns RiskInfo::green unless availableMargin is lower than "
         "initial margin requirements ",
         py::arg("amount_to_purchase"))
    .def("checkRisk", (RiskInfo (Portfolio::*) (int, double) const) &Portfolio::checkRisk,
         "checks availableMargin and compares with proposed amount_to_purchase"
         "I.e returns RiskInfo::green unless availableMargin is lower than "
         "initial margin requirements ",
         py::arg("assetIdx"), py::arg("units"))
    .def("checkRisk", (RiskInfo (Portfolio::*) (string, double) const) &Portfolio::checkRisk,
         "checks availableMargin and compares with proposed amount_to_purchase"
         "I.e returns RiskInfo::green unless availableMargin is lower than "
         "initial margin requirements ",
         py::arg("assetCode"), py::arg("units"))
    .def("handleTransaction", (void(Portfolio::*)(int, double, double, double))
         &Portfolio::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("assetIdx"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0.)
    .def("handleTransaction", (void(Portfolio::*)(string, double, double, double))
         &Portfolio::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("asset"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0.)
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

    // .def(py::init<>())
  _Account.def(py::init<Portfolio&> (), py::arg("portfolio"))
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
    .def("setRequiredMargin", (void (Account::*)(double)) &Account::setRequiredMargin,
         "set required Margin level for default port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("requiredMargin"))
    .def("setRequiredMargin", (void (Account::*)(string, double)) &Account::setRequiredMargin,
         "set required Margin level for specified port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("portID"), py::arg("requiredMargin"))
    .def("setMaintenanceMargin", (void (Account::*)(double)) &Account::setMaintenanceMargin,
         "set maintenance Margin level for default port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("maintenanceMargin"))
    .def("setMaintenanceMargin", (void (Account::*)(string, double)) &Account::setMaintenanceMargin,
         "set maintenance Margin level for specified port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("portID"), py::arg("maintenanceMargin"))
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
    .def("handleTransaction", (void(Account::*)(int, double, double, double))
         &Account::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("assetIdx"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0.)
    .def("handleTransaction", (void(Account::*)(string, double, double, double))
         &Account::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("asset"), py::arg("transactionPrice"), py::arg("units"),
         py::arg("transactionCost")=0.)
    .def("handleTransaction", (void(Account::*)(string, int, double, double, double))
         &Account::handleTransaction,
         "handle transaction given asset idx, transactionPrice, and amount of units",
         py::arg("portID"), py::arg("assetIdx"), py::arg("transactionPrice"),
         py::arg("units"), py::arg("transactionCost")=0.)
    .def("handleTransaction", (void(Account::*)(string, string, double, double, double))
         &Account::handleTransaction,
         "handle transaction given asset code str, transactionPrice, and amount of units",
         py::arg("pordID"), py::arg("asset"), py::arg("transactionPrice"),
         py::arg("units"), py::arg("transactionCost")=0.)
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


  _Broker.def(py::init<Account&> (), py::arg("account"))
    .def(py::init<Portfolio&> (), py::arg("portfolio"))
    .def(py::init<Assets, double> (), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def(py::init<string, Assets, double> (),
         py::arg("AccId"), py::arg("assets"), py::arg("initCash")=double(1'000'000))
    .def("setDataSource", &Broker::setDataSource,
         "set data source to query/link current prices to",
         py::arg("DataSource object"))
    .def("addAccount", &Broker::addAccount,
         "add account by copying given account object")
    .def("addPortfolio", (void (Broker::*)(const Portfolio&)) &Broker::addPortfolio,
         "add port to default account by copying given port object",
         py::arg("port"))
    .def("addPortfolio", (void (Broker::*)(string, const Portfolio&)) &Broker::addPortfolio,
         "add port to account specified by accID by copying given port object",
         py::arg("accID"), py::arg("port"))
    .def("setRequiredMargin", (void (Broker::*)(double)) &Broker::setRequiredMargin,
         "set requiredMargin level for default Acc"
        "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("requiredMarginLevel"))
    .def("setRequiredMargin", (void (Broker::*)(string, double)) &Broker::setRequiredMargin,
         "set requiredMargin level for specified Acc"
         "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("accID"), py::arg("requiredMarginLevel"))
    .def("setRequiredMargin", (void (Broker::*)(string, string, double)) &Broker::setRequiredMargin,
         "set requiredMargin level for specified  port in specified Acc"
         "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("accID"), py::arg("portID"), py::arg("requiredMarginLevel"))
    .def("setMaintenanceMargin", (void (Broker::*)(double)) &Broker::setMaintenanceMargin,
         "set maintenanceMargin level for default Acc"
         "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("maintenanceMarginLevel"))
    .def("setMaintenanceMargin", (void (Broker::*)(string, double)) &Broker::setMaintenanceMargin,
         "set maintenanceMargin level for specified Acc"
         "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("accID"), py::arg("maintenanceMarginLevel"))
    .def("setMaintenanceMargin", (void (Broker::*)(string, string, double)) &Broker::setMaintenanceMargin,
         "set maintenanceMargin level for specified  port in specified Acc"
         "takes margin level as a proprtion as input "
         "I.e 0.1 for 10% reuqired margin or 10x levarage",
         py::arg("accID"), py::arg("portID"), py::arg("requiredMarginLevel"))
    .def("account", (const Account& (Broker::*)() const) &Broker::account,
         "Returns default account - useful if just one acc",
         py::return_value_policy::reference)
    .def("account", (const Account& (Broker::*)(string) const) &Broker::account,
         "Returns acc by accID. Throws out_of_range error if accID doesn't exist",
         py::return_value_policy::reference)
    .def("accounts", (std::vector<Account>&(Broker::*)() const) &Broker::accounts,
         "Return list of accounts",
         py::return_value_policy::reference)
    .def("portfolios", (std::vector<Portfolio>(Broker::*)() const) &Broker::portfolios,
         "Return list of portfolios",
         py::return_value_policy::move)
    .def("portfolios", (const std::vector<Portfolio>&(Broker::*)(string accID) const) &Broker::portfolios,
         "Return list of portfolios", py::arg("accID"),
         py::return_value_policy::reference)
    .def("portfolio", (const Portfolio&(Broker::*)() const) &Broker::portfolio,
         "return default portfolio",
         py::return_value_policy::copy)
    .def("portfolio", (const Portfolio&(Broker::*)(string portID) const) &Broker::portfolio,
         "return portfolio given specified ID", py::arg("portID"),
         py::return_value_policy::reference)
    .def("portfolioBook", (std::unordered_map<string, Portfolio> (Broker::*)() const)
         &Broker::portfolioBook,
         "Return dict of portfolios for all accounts",
         py::return_value_policy::move)
    .def("portfolioBook", (const PortfolioBook& (Broker::*)(string accID) const)
         &Broker::portfolioBook,
         "Return dict of portfolios for specific acc", py::arg("accID"),
         py::return_value_policy::reference)
    .def("handleAction", (BrokerResponseMulti (Broker::*)(AmountVector& units) )
         &Broker::handleAction,
         "handle a transaction as per the given vector of units to be purchased",
         py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(int, double) )
         &Broker::handleTransaction,
         "handle a transaction given asset index and units to purchase",
         py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(string, double) )
         &Broker::handleTransaction,
         "handle a transaction given asset code and units to purchase",
         py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(string, int, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, asset index and units to purchase",
         py::arg("accID"), py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(string, string, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, asset code and units to purchase",
         py::arg("accID"), py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(string, string, int, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, portID, asset index and units to purchase",
         py::arg("accID"), py::arg("portID"), py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (std::pair<double, double>(Broker::*)(string, string, string, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, portID, asset index and units to purchase",
         py::arg("accID"), py::arg("portID"), py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move);

  _Env.def(py::init<string, Assets, double> (),
         py::arg("dataSourceType"),
         py::arg("assets"),
         py::arg("initCash")=double(1'000'000))
    .def(py::init<string, Assets, double, py::dict> (),
         py::arg("dataSourceType"),
         py::arg("assets"),
         py::arg("initCash"),
         py::arg("config_dict"))
    .def("step", (SRDI<double> (Env::*)()) &Env::step,
         "take env step with no action");

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
