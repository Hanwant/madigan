#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "DataSource.h"
#include "PyDataSource.h"
#include "Portfolio.h"
#include "Account.h"
#include "Broker.h"
#include "Env.h"
#include "PyEnv.h"
#include "Config.h"

namespace py = pybind11;
using namespace madigan;

// PYBIND11_MAKE_OPAQUE(PortfolioBook);
// PYBIND11_MAKE_OPAQUE(std::unordered_map<string, Portfolio>);
// PYBIND11_MAKE_OPAQUE(vector<Portfolio>);
// PYBIND11_MAKE_OPAQUE(std::tuple);
// PYBIND11_MAKE_OPAQUE(SRDI<double>);
// PYBIND11_MAKE_OPAQUE(SRDI<PriceVector>);
// PYBIND11_MAKE_OPAQUE(SRDISingle);
// PYBIND11_MAKE_OPAQUE(SRDIMulti);
// PYBIND11_MAKE_OPAQUE(std::tuple<State, double, bool, EnvInfo<double>>);
// PYBIND11_MAKE_OPAQUE(std::tuple<State, double, bool, EnvInfo<PriceVector>>);

// using SRDI = std::tuple<State, double, bool, EnvInfo<T>>;

template<typename T>
void declareBrokerResponse(py::module& m, const string& className){
  using Class = BrokerResponse<T>;
  py::class_<Class>(m, className.c_str())
    .def_readonly("event", &Class::event)
    .def_readonly("timestamp", &Class::timestamp)
    .def_readonly("transactionPrice", &Class::transactionPrice)
    .def_readonly("transactionUnits", &Class::transactionUnits)
    .def_readonly("transactionCost", &Class::transactionCost)
    .def_readonly("riskInfo", &Class::riskInfo)
    .def_readonly("marginCall", &Class::marginCall)
    .def("__repr__", [](const BrokerResponse<T>& b){
        std::stringstream repr;
        repr << "timestamp:        " << b.timestamp << "\n";
        repr << "transactionPrice: \n" << b.transactionPrice << "\n";
        repr << "transactionUnits:  \n" << b.transactionUnits << "\n";
        repr << "transactionCost:  \n" << b.transactionCost << "\n";
        repr << "riskInfo:         \n" << b.riskInfo << "\n";
        repr << "marginCall:         \n" << b.marginCall<< "\n";
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

  // Declared and defined in the same order
  py::enum_<RiskInfo> _RiskInfo(m, "RiskInfo");
  declareEnvInfo<double>(m, "EnvInfoSingle");
  declareEnvInfo<PriceVector>(m, "EnvInfoMulti");
  declareBrokerResponse<double>(m, "BrokerResponseSingle");
  declareBrokerResponse<PriceVector>(m, "BrokerResponseMulti");
  py::class_<State> _State(m, "State");
  // py::class_<SRDI<double>> _SRDISingle(m, "SRDISingle");
  // py::class_<SRDI<PriceVector>> _SRDIMulti(m, "SRDIMulti");

  py::class_<Asset> _Asset(m, "Asset");
  py::class_<Assets> _Assets(m, "Assets");
  py::class_<PriceVector> _PriceVector(m, "PriceVector", py::buffer_protocol());
  py::bind_map<std::unordered_map<string, Portfolio>>(m, "PortfolioBook");
  py::bind_map<std::unordered_map<string, Account>>(m, "AccountBook");

  py::class_<DataSourceTick, PyDataSource>_DataSourceTick(m, "DataSourceTick");
  py::class_<Synth, DataSourceTick>_Synth(m, "Synth");
  py::class_<SawTooth, Synth>_SawTooth(m, "SawTooth");
  py::class_<Triangle, Synth>_Triangle(m, "Triangle");
  py::class_<SineAdder, DataSourceTick>_SineAdder(m, "SineAdder");
  py::class_<SineDynamic, DataSourceTick>_SineDynamic(m, "SineDynamic");
  py::class_<SimpleTrend, DataSourceTick>_SimpleTrend(m, "SimpleTrend");
  py::class_<TrendOU, DataSourceTick>_TrendOU(m, "TrendOU");
  py::class_<TrendyOU, DataSourceTick>_TrendyOU(m, "TrendyOU");
  py::class_<Composite, DataSourceTick>_Composite(m, "Composite");
  py::class_<HDFSource, DataSourceTick, PyHDFSource>_HDFSource(m, "HDFSource");

  py::class_<Portfolio>_Portfolio(m, "Portfolio");
  py::class_<Account>_Account(m, "Account");
  py::class_<Broker> _Broker(m, "Broker");
  py::class_<Env, PyEnv> _Env(m, "Env");

  _RiskInfo.value("green", RiskInfo::green)
    .value("margin_call", RiskInfo::margin_call)
    .value("insuff_margin", RiskInfo::insuff_margin)
    .value("blown_out", RiskInfo::blown_out)
    .def("__str__", [] (const RiskInfo& r) {
      std::stringstream repr;
      repr << r;
      return repr.str(); })
    .def("__repr__", [] (const RiskInfo& r){
      std::stringstream repr;
      repr << r;
      return repr.str(); });

  _State
    .def(py::init<PriceVector, Ledger, std::size_t> (),
         py::arg("prices"), py::arg("portfolio"), py::arg("timestamp"))
    .def_readwrite("price", &State::price)
    .def_readwrite("portfolio", &State::portfolio)
    .def_readwrite("timestamp", &State::timestamp);

  // _SRDISingle
  //   .def(py::init<State, double, bool, EnvInfo<double>> (),
  //        py::arg("State"), py::arg("reward"),
  //        py::arg("done"), py::arg("EnvInfoSingle"))
  //   .def_property_readonly("State", [](const SRDISingle& srdi){return std::get<0>(srdi);})
  //   .def_property_readonly("reward", [](const SRDISingle& srdi){return std::get<1>(srdi);})
  //   .def_property_readonly("done", [](const SRDISingle& srdi){return std::get<2>(srdi);})
  //   .def_property_readonly("EnvInfoSingle", [](const SRDISingle& srdi){return std::get<3>(srdi);});

  _Asset.def(py::init<string> (), py::arg("asset_name"))
    .def(py::init<string, string> (), py::arg("asset_name"), py::arg("exchange"))
    .def_readwrite("name", &Asset::name)
    .def_readwrite("code", &Asset::code)
    .def_readwrite("exchange", &Asset::exchange)
    .def_readwrite("bp_multiplier", &Asset::bpMultiplier)
    .def("__str__", [] (const Asset& a) { return a.code; })
    .def("__repr__", [] (const Asset& a){ return a.code; });

  _Assets.def(py::init<std::vector<string>> (), py::arg("asset_names_list"))
    .def(py::init<std::vector<Asset>> (), py::arg("asset_names_list"))
    .def("__len__", [](const Assets &v) { return v.size(); })
    .def("__repr__", [](const Assets &v) {
      std::stringstream repr;
      repr << "[";
      if(v.size() > 0){
        for (int i=0; i<v.size()-1; i++){
          repr << v[i].name << ", ";
        }
        repr << v.back().name;
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
         double, double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"), py::arg("noise"))
    .def_property_readonly("currentTime", &Synth::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (Synth::*) ()) &Synth::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (Synth::*) ()) &Synth::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (Synth::*) ()) &Synth::currentData,
         "Get current data - make be raw prices or preprocessed or anything else",
         py::return_value_policy::reference);
  _SawTooth.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         double, double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"), py::arg("noise"))
    .def_property_readonly("currentTime", &SawTooth::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (SawTooth::*) ()) &SawTooth::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (SawTooth::*) ()) &SawTooth::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _Triangle.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         double, double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"), py::arg("noise"))
    .def_property_readonly("currentTime", &Triangle::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (Triangle::*) ()) &Triangle::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (Triangle::*) ()) &Triangle::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _SineAdder.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         double, double> (),
         py::arg("freq"), py::arg("mu"),
         py::arg("amp"), py::arg("phase"),
         py::arg("dx"), py::arg("noise"))
    .def_property_readonly("currentTime", &SineAdder::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (SineAdder::*) ()) &SineAdder::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (SineAdder::*) ()) &SineAdder::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (SineAdder::*) ()) &SineAdder::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _SineDynamic.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<std::array<double, 3>>, vector<std::array<double, 3>>,
         vector<std::array<double, 3>>, double, double> (),
         py::arg("freqRange"), py::arg("muRange"),
         py::arg("ampRange"), py::arg("dx"), py::arg("noise"))
    .def_property_readonly("currentTime", &SineDynamic::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (SineDynamic::*) ()) &SineDynamic::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (SineDynamic::*) ()) &SineDynamic::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("getProcess", &SineDynamic::getProcess,
         "get wavetable data directly", py::arg("i"))
    .def("currentData", (PriceVector& (SineDynamic::*) ()) &SineDynamic::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _SimpleTrend.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<int>,
         vector<int>, vector<double>,
         vector<double>, vector<double>,
         vector<double>> (),
         py::arg("trend_prob"), py::arg("min_period"),
         py::arg("max_period"), py::arg("noise"),
         py::arg("dYMin"), py::arg("dYMax"), py::arg("start"))

    .def_property_readonly("currentTime", &SimpleTrend::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (SimpleTrend::*) ()) &SimpleTrend::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (SimpleTrend::*) ()) &SimpleTrend::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (SimpleTrend::*) ()) &SimpleTrend::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _TrendOU.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<int>,
         vector<int>, vector<double>,
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         vector<double>, vector<double>> (),
         py::arg("trend_prob"), py::arg("min_period"),
         py::arg("max_period"), py::arg("dYMin"),
         py::arg("dYMax"), py::arg("start"),
         py::arg("theta"), py::arg("phi"),
         py::arg("noise_trend"), py::arg("ema_alpha"))
    .def_property_readonly("currentTime", &TrendOU::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (TrendOU::*) ()) &TrendOU::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (TrendOU::*) ()) &TrendOU::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (TrendOU::*) ()) &TrendOU::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _TrendyOU.def(py::init<>())
    .def(py::init<py::dict>(), py::arg("config_dict"))
    .def(py::init<
         vector<double>, vector<int>,
         vector<int>, vector<double>,
         vector<double>, vector<double>,
         vector<double>, vector<double>,
         vector<double>, vector<double>> (),
         py::arg("trend_prob"), py::arg("min_period"),
         py::arg("max_period"), py::arg("dYMin"),
         py::arg("dYMax"), py::arg("start"),
         py::arg("theta"), py::arg("phi"),
         py::arg("noise_trend"), py::arg("ema_alpha"))
    .def_property_readonly("currentTime", &TrendyOU::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (TrendyOU::*) ()) &TrendyOU::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (TrendyOU::*) ()) &TrendyOU::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("currentData", (PriceVector& (TrendyOU::*) ()) &TrendyOU::currentData,
         "Get current data points",
         py::return_value_policy::reference);
  _Composite.def(py::init<py::dict>(), py::arg("config_dict"))
    .def_property_readonly("nAssets", &Composite::nAssets,
                           "number of Assets - length of currentPrices",
                           py::return_value_policy::move)
    .def_property_readonly("currentTime", &Composite::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (Composite::*) ()) &Composite::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (Composite::*) ()) &Composite::currentPrices,
         "Get current prices",
         py::return_value_policy::reference)
    .def("dataSources", [](const Composite& obj){
      auto pylist = py::list();
      for (auto& ptr: obj.dataSources()){
        auto pyObj = py::cast(*ptr, py::return_value_policy::reference);
        pylist.append(pyObj);
      }
      return pylist;
    },
         "Returns reference to vector of unique_ptr<DataSource>",
         py::return_value_policy::copy)
    .def("currentData", (PriceVector& (Composite::*) ()) &Composite::currentData,
         "Get current data - make be raw prices or preprocessed or anything else",
         py::return_value_policy::reference);
  _HDFSource.def(py::init<string, string, string, string>(),
                 py::arg("filepath"), py::arg("mainKey"),
                 py::arg("priceKeu"), py::arg("timestampKey"))
    .def_property_readonly("nAssets", &HDFSource::nAssets,
                           "number of Assets - length of currentPrices",
                           py::return_value_policy::move)
    .def_property_readonly("currentTime", &HDFSource::currentTime,
                           "get the current timestamp",
                           py::return_value_policy::move)
    .def("getData", (PriceVector& (HDFSource::*) ()) &HDFSource::getData,
         "Get Next data points",
         py::return_value_policy::reference)
    .def("currentPrices", (PriceVector& (HDFSource::*) ()) &HDFSource::currentPrices,
         "Get current prices",
         py::return_value_policy::reference);

  _Portfolio.def(py::init<string, Assets, double> (),
         py::arg("id"),
         py::arg("assets"),
         py::arg("initCash")=double(1'000'000))
    .def("setDataSource", &Portfolio::setDataSource,
         "assign data source for current prices reference")
    .def_property_readonly("id", &Portfolio::id, "portfolio id")
    .def_property_readonly("initCash", &Portfolio::initCash,
                           "initial amount of deposited cash")
    .def_property_readonly("requiredMargin", &Portfolio::requiredMargin,
                           "required margin level")
    .def_property_readonly("maintenanceMargin", &Portfolio::maintenanceMargin,
                           "maintenance margin level")
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
    .def_property_readonly("pnlPositions", &Portfolio::pnl,
                           "returns current net profit/loss for each position",
                           py::return_value_policy::move)
    .def_property_readonly("borrowedMargin", &Portfolio::borrowedMargin,
                           "Total Margin currently borrowed for positions (i.e for levaraged buy)")
    .def_property_readonly("borrowedMarginLedger", &Portfolio::borrowedMarginLedger,
                           "Margin currently borrowed for each position (i.e for levaraged buy)")
    .def_property_readonly("availableMargin", &Portfolio::availableMargin,
                           "Margin available for entering positions (i.e for levaraged buy)")
    .def_property_readonly("assetValue", &Portfolio::assetValue,
                           "net value of current positions (long + short)")
    .def_property_readonly("positionValues", &Portfolio::positionValues,
                           "vector of position values for current holdings",
                           py::return_value_policy::move)
    .def_property_readonly("positionValuesFull", &Portfolio::positionValuesFull,
                           "vector of position values for current holdings, "
                           " including cash holdings",
                           py::return_value_policy::move)
    .def_property_readonly("borrowedAssetValue", &Portfolio::borrowedAssetValue,
                           "value of current short positions")
    .def_property_readonly("usedMargin", &Portfolio::usedMargin,
                           "Total margin currently committed to positions ")
    .def_property_readonly("usedMarginLedger", &Portfolio::usedMarginLedger,
                           "Ledger showing margin currently committed to positions. "
                           "Short positions will be negative")
    .def_property_readonly("currentPrices", &Portfolio::currentPrices,
                           "current prices as per registered data source "
                           "\n Careful as the returned array will contain a reference"
                           "to the datasource buffer - const is ignored",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND CONNECTED TO DATA SOURCE*/
    .def_property_readonly("meanEntryPrices", &Portfolio::meanEntryPrices,
                           "Mean Entry price for the current positions "
                           "\n Careful as the returned array will contain a reference"
                           "to the internal array- const is ignored",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND CONNECTED TO DATA SOURCE*/
    .def_property_readonly("meanEntryValue", &Portfolio::meanEntryValue,
                           "Mean Entry Value for the current positions ",
                           py::return_value_policy::move)
    .def_property_readonly("ledger", &Portfolio::ledger,
                           "vector of asset holdings",
                           py::return_value_policy::reference)
    .def_property_readonly("ledgerFull", &Portfolio::ledgerFull,
                           "returns full ledger, includes cash as well as asset holdings",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerNormed", &Portfolio::ledgerNormed,
                           "returns current ledger "
                           " normalized by equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerNormedFull", &Portfolio::ledgerNormedFull,
                           "returns current ledger along with cash holdings"
                           " (- borrowedMargin)"
                           " normalized by total equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerAbsNormed", &Portfolio::ledgerAbsNormed,
                           "returns current ledger for the default portfolio"
                           " normalized by abs position sizes and cash balance"
                           " prevents numerical instability when using equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerAbsNormedFull", &Portfolio::ledgerAbsNormedFull,
                           "returns current ledger along with cash holdings"
                           " (- borrowedMargin)"
                           " normalized by abs position sizes and cash balance"
                           " prevents numerical instability when using equity",
                           py::return_value_policy::move)
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
    .def("close", (void(Portfolio::*)(int , double, double))
         &Portfolio::close,
         "Close Position for given asset Idx, if a position exists.",
         py::arg("assetIdx"), py::arg("transactionPrice"),
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
    .def("setSlippage", (void (Broker::*)(double, double)) &Broker::setSlippage,
         "set relative and absolute slippage- for all accounts/portfolios",
         py::arg("relativeSlippage"), py::arg("absSlippage"))
    .def("setTransactionCost", (void (Broker::*)(double, double)) &Broker::setTransactionCost,
         "set relative and absolute transaction costs - for all accounts/portfolios",
         py::arg("relativeCost"), py::arg("absCost"))
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
    .def("handleAction", (BrokerResponseMulti (Broker::*)(const AmountVector& units) )
         &Broker::handleAction,
         "handle a transaction as per the given vector of units to be purchased",
         py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseMulti (Broker::*)(const AmountVector& units) )
         &Broker::handleTransaction,
         "handle a vector of desired units to purchase"
         "Each unit is treated as a separate transaction with its own risk profile",
         py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseMulti (Broker::*)(string, const AmountVector&) )
         &Broker::handleTransaction,
         "handle a vector of desired units to purchase"
         "Each unit is treated as a separate transaction with its own risk profile",
         py::arg("accID"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseMulti (Broker::*)(string, string, const AmountVector&) )
         &Broker::handleTransaction,
         "handle a vector of desired units to purchase"
         "Each unit is treated as a separate transaction with its own risk profile",
         py::arg("accID"), py::arg("portID"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(int, double) )
         &Broker::handleTransaction,
         "handle a transaction given asset index and units to purchase",
         py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(string, double) )
         &Broker::handleTransaction,
         "handle a transaction given asset code and units to purchase",
         py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(string, int, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, asset index and units to purchase",
         py::arg("accID"), py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(string, string, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, asset code and units to purchase",
         py::arg("accID"), py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(string, string, int, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, portID, asset index and units to purchase",
         py::arg("accID"), py::arg("portID"), py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("handleTransaction", (BrokerResponseSingle(Broker::*)(string, string, string, double) )
         &Broker::handleTransaction,
         "handle a transaction given accID, portID, asset index and units to purchase",
         py::arg("accID"), py::arg("portID"), py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("close", (BrokerResponseSingle(Broker::*)(int))
         &Broker::close,
         "Close Position for given asset Idx, if a position exists. "
         "Applies slippage and transaction cost etc as normal",
         py::arg("assetIdx"))
    .def("handleAction", (BrokerResponseMulti(Broker::*)(const AmountVector&) )
         &Broker::handleAction,
         "Routing to handleTransaction given array of units "
         "Provides env api for RL based framework",
         py::arg("units"),
         py::return_value_policy::move)
    .def("handleEvent", (BrokerResponseMulti(Broker::*)(const AmountVector&) )
         &Broker::handleEvent,
         "Routing to handleTransaction given array of units "
         "Provides api for event based framework",
         py::arg("units"),
         py::return_value_policy::move);

  _Env.def(py::init<string, double> (),
         py::arg("dataSourceType"),
         py::arg("initCash")=double(1'000'000))
    .def(py::init<string, double, py::dict> (),
         py::arg("dataSourceType"),
         py::arg("initCash"),
         py::arg("config_dict"))
    .def("reset", &Env::reset,
         "reset dataSource and env",
         py::return_value_policy::move)
    .def("setDataSource", (void (Env::*)(DataSourceTick*)) &Env::setDataSource,
         "set datasource from python instance subclassing DataSource",
         py::arg("dataSource"))
    .def("setRequiredMargin", (void (Env::*)(double)) &Env::setRequiredMargin,
         "set required Margin level for default port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("requiredMargin"))
    .def("setMaintenanceMargin", (void (Env::*)(double)) &Env::setMaintenanceMargin,
         "set maintenance Margin level for default port, takes proportion as input"
         " I.e 0.1 for 10% margin or 10x levarage",
         py::arg("maintenanceMargin"))
    .def("setSlippage", (void (Env::*)(double, double)) &Env::setSlippage,
         "set relative and absolute slippage- for all accounts/portfolios",
         py::arg("relativeSlippage"), py::arg("absSlippage"))
    .def("setTransactionCost", (void (Env::*)(double, double)) &Env::setTransactionCost,
         "set relative and absolute transaction costs - for all accounts/portfolios",
         py::arg("relativeCost"), py::arg("absCost"))
    // .def_property_readonly("initCash", &Env::defaultPortfolio::initCash,
    //                        "initCash for the default portfolio")
    .def_property_readonly("broker", &Env::broker,
                           "Returns pointer to broker instance")
    .def_property_readonly("account", (const Account* (Env::*)() const) &Env::account,
                           "Returns pointer to default account instance")
    .def_property_readonly("portfolio",(const Portfolio* (Env::*)() const) &Env::portfolio,
                           "Returns pointer to default portfolio instance")
    .def_property_readonly("dataSource", &Env::dataSource,
                           "returns reference to current buffer for prices"
                           ", use reference with care as constness is cast away on python side")
    .def_property_readonly("isDateTime", &Env::isDateTime,
                           "indicates whether the timestamps from the dataSource"
                           "are in datetime units (sec past epoch) or a normal index")
    .def_property_readonly("requiredMargin", &Env::requiredMargin,
                           "required margin level")
    .def_property_readonly("maintenanceMargin", &Env::maintenanceMargin,
                           "maintenance margin level")

    .def_property_readonly("timestamp", &Env::currentTime,
                           "Alias for currentTime, returns current timestamp"
                           "as an integer as per dataSource",
                           py::return_value_policy::copy)
    .def_property_readonly("currentTime", &Env::currentTime,
                           "returns current timestamp as an integer as per dataSource",
                           py::return_value_policy::copy)

    .def_property_readonly("currentPrices", &Env::currentPrices,
                           "returns reference to current buffer for prices"
                           ", use reference with care as constness is cast away on python side",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND CONNECTED TO DATA SOURCE*/

    .def_property_readonly("meanEntryPrices", &Env::meanEntryPrices,
                           "Mean Entry price for the current positions "
                           "returns reference to current buffer for prices"
                           ", use reference with care as constness is cast away on python side",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND CONNECTED TO DATA SOURCE*/
    .def_property_readonly("ledger", &Env::ledger,
                           "returns reference to current ledger for the default portfolio"
                           ", use reference with care as constness is cast away on python side",
                           py::return_value_policy::reference)/* BE CAREFUL - CASTS AWAY CONSTNESS
                                                                 AND RETURNS REFERENCE */
    .def_property_readonly("ledgerFull", &Env::ledgerFull,
                           "returns full ledger, includes cash as well as asset holdings",
                           " for the default portfolio.",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerNormed", &Env::ledgerNormed,
                           "returns current ledger for the default portfolio"
                           " normalized by position sizes and cash balance - I.e equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerAbsNormed", &Env::ledgerAbsNormed,
                           "returns current ledger for the default portfolio"
                           " normalized by abs position sizes and cash balance"
                           " prevents numerical instability when using equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerNormedFull", &Env::ledgerNormedFull,
                           "returns current ledger along with cash holdings"
                           " (- borrowedMargin)"
                           " normalized by total equity",
                           py::return_value_policy::move)
    .def_property_readonly("ledgerAbsNormedFull", &Env::ledgerAbsNormedFull,
                           "returns current ledger along with cash holdings"
                           " (- borrowedMargin)"
                            " normalized by abs position sizes and cash balance"
                            " prevents numerical instability when using equity",
                           py::return_value_policy::move)
    .def_property_readonly("equity",  &Env::equity,
                           "returns net equity")
    .def_property_readonly("cash",  &Env::cash,
                           "returns net cash")
    .def_property_readonly("assetValue",  &Env::assetValue,
                           "returns net value of assets (incl lnog and short)")
    .def_property_readonly("positionValues",  &Env::positionValues,
                           "returns values of individual assets currently held",
                           py::return_value_policy::move)
    .def_property_readonly("positionValuesFull", &Env::positionValuesFull,
                           "vector of position values for current holdings, "
                           " including cash holdings",
                           py::return_value_policy::move)
    .def_property_readonly("pnl",  &Env::pnl,
                           "returns net pnl of all positions")
    .def_property_readonly("pnlPositions",  &Env::pnlPositions,
                           "returns individual pnls of positions",
                           py::return_value_policy::move)
    .def_property_readonly("usedMargin", &Env::usedMargin,
                           "returns net usedMargin")
    .def_property_readonly("availableMargin", &Env::availableMargin,
                           "returns net availableMargin")
    .def_property_readonly("borrowedMargin", &Env::borrowedMargin,
                           "returns net borrowedMargin")
    .def_property_readonly("borrowedAssetValue", &Env::borrowedAssetValue,
                           "returns net borrowed assets (i.e value of shorts)")
    .def_property_readonly("nAssets", &Env::nAssets,
                           "number of assets")
    .def_property_readonly("assets", &Env::assets,
                           "Returns list of Asset objects")
    .def("equity_", (double(Env::*)(string) const) &Env::equity,
         "returns net equity")
    .def("cash_", (double(Env::*)(string) const) &Env::cash,
         "returns net cash")
    .def("assetValue_", (double(Env::*)(string) const) &Env::assetValue,
         "returns net assetValue (long and short)")
    .def("usedMargin_", (double(Env::*)(string) const) &Env::usedMargin,
         "returns net usedMargin")
    .def("availableMargin_", (double(Env::*)(string) const) &Env::availableMargin,
         "returns net availableMargin")
    .def("borrowedMargin_", (double(Env::*)(string) const) &Env::borrowedMargin,
         "returns net borrowedMargin for specific portID")
    .def("borrowedAssetValue_", (double(Env::*)(string) const) &Env::borrowedAssetValue,
         "returns net borrowedAssetValue for specific portID")
    .def("checkRisk", &Env::checkRisk,
         "Checks for margin call, useful as when attempting to reverse a position, "
         "a margin call will not be returned by checkRisk(asset, units) as "
         "the transaction allowing closing of the positions will take precendence"
         "in the return RiskInfo")
    .def("step", (SRDIMulti (Env::*)(const AmountVector&)) &Env::step,
         "take env step with given array of units to purchase",
         py::arg("units"),
         py::return_value_policy::move)
    .def("step", (SRDISingle (Env::*)(int, double)) &Env::step,
         "take env step with given array of units to purchase",
         py::arg("assetIdx"), py::arg("units"),
         py::return_value_policy::move)
    .def("step", (SRDISingle (Env::*)(string, double)) &Env::step,
         "take env step with given array of units to purchase",
         py::arg("assetCode"), py::arg("units"),
         py::return_value_policy::move)
    .def("step", (SRDISingle (Env::*)()) &Env::step,
         "take env step with no action",
         py::return_value_policy::move);
    // .def("__copy__", [](const Env &self) {
    //   return Env(self.dataSourceType(), self.assets(), self.initCash(), self.config);
    // })
    // .def("__deepcopy__", [](const Env &self, py::dict memo) {
    //   return Env(self.dataSourceType(), self.assets(), self.initCash(), self.config);
    // });

}

//======================================================================
//=========== Testing Passing dict/map data to and from python =========
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
