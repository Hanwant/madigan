#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <memory>
#include <unordered_map>
#include <any>
#include <type_traits>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace madigan{

  using std::string;
  using std::vector;

  // ================================================================
  // ========= Data Types implicitly converitble to numpy ===========
  // ================================================================
  typedef Eigen::VectorXi ActionVector;
  typedef Eigen::VectorXd PriceVector;
  typedef Eigen::Map<const PriceVector> PriceVectorMap;
  typedef Eigen::VectorXd AmountVector;
  typedef Eigen::VectorXd Ledger;
  typedef Eigen::Map<const Ledger> LedgerMap;

  // ================================================================
  // ================== Custom Exceptions for project ===============
  // ================================================================
  class ConfigError: public std::logic_error
  {
  public:
    ConfigError(std::string message) : std::logic_error(message) { };
  };

  class NotImplemented : public std::logic_error
  {
  public:
    NotImplemented(std::string message) : std::logic_error(message) { };
  };

  // ================================================================
  // ======================== Data Containers =======================
  // ================================================================

  struct State{
    PriceVector price;
    Ledger portfolio;
    std::size_t timestamp;
    State(PriceVector price, Ledger portfolio):
      price(price), portfolio(portfolio), timestamp(0){}
    State(Eigen::Ref<const PriceVector> price, Eigen::Ref<const Ledger> portfolio):
      price(price), portfolio(portfolio), timestamp(0){}
    State(Eigen::Ref<const PriceVector> price, Eigen::Ref<const Ledger> portfolio,
          std::size_t timestamp):
      price(price), portfolio(portfolio), timestamp(timestamp){}
  };

  struct DataItem{
    PriceVector prices;
    // TimeStamp timestamp;
  };

  enum class RiskInfo {
    green,
    insuff_margin,
    margin_call,
    blown_out
  };

  inline std::ostream& operator<<(std::ostream& os, RiskInfo risk){
    switch(risk)
      {
      case RiskInfo::green: os << "green"; break;
      case RiskInfo::insuff_margin: os << "insuff_margin"; break;
      case RiskInfo::margin_call: os << "margin_call"; break;
      case RiskInfo::blown_out: os << "blown_out"; break;
      }
    return os;
  }

  inline std::ostream& operator<<(std::ostream& os, std::vector<RiskInfo> risk){
    for (const auto& _risk: risk){
      switch(_risk)
        {
        case RiskInfo::green: os << "green"; break;
        case RiskInfo::insuff_margin: os << "insuff_margin"; break;
        case RiskInfo::margin_call: os << "margin_call"; break;
        case RiskInfo::blown_out: os << "blown_out"; break;
        }
      os << ", ";
    }
    return os;
  }

  // struct BrokerResponseBase;
  template<typename T>
  struct BrokerResponse {
    std::string event;
    T transactionPrice;
    T transactionUnits;
    T transactionCost;
    std::size_t timestamp;
    bool marginCall{false};
    typedef typename std::conditional<std::is_same<T, PriceVector>::value,
                                      std::vector<RiskInfo>,
                                      RiskInfo>::type RiskInfoType;
    RiskInfoType riskInfo;

    BrokerResponse(){};

    BrokerResponse(std::string event, T transPrice, T transUnits, T transCost):
      event(event), transactionPrice(transPrice), transactionUnits(transUnits),
      transactionCost(transCost)
    {}
    BrokerResponse(std::string event, T transPrice, T transUnits, T transCost, RiskInfoType riskInfo):
      event(event), transactionPrice(transPrice), transactionUnits(transUnits),
      transactionCost(transCost), riskInfo(riskInfo){}
    BrokerResponse(std::string event, T transPrice, T transUnits, T transCost, RiskInfoType riskInfo,
                   bool marginCall):
      event(event), transactionPrice(transPrice), transactionUnits(transUnits),
      transactionCost(transCost), riskInfo(riskInfo), marginCall(marginCall){}

    BrokerResponse(T transPrice, T transUnits, T transCost):
      BrokerResponse("", transPrice, transUnits, transCost){}
    BrokerResponse(T transPrice, T transUnits, T transCost, RiskInfoType riskInfo):
      BrokerResponse("", transPrice, transUnits, transCost, riskInfo){}
    BrokerResponse(T transPrice, T transUnits, T transCost, RiskInfoType riskInfo, bool marginCall):
      BrokerResponse("", transPrice, transUnits, transCost, riskInfo, marginCall){}
  };

  using BrokerResponseSingle = BrokerResponse<double>;
  using BrokerResponseMulti = BrokerResponse<PriceVector>;

  template<typename T>
  struct EnvInfo{
    // std::unique_ptr<BrokerResponseBase> brokerResponse;
    BrokerResponse<T> brokerResponse;
    EnvInfo(){};
    // EnvInfo(std::unique_ptr<BrokerResponse> brokerResp): brokerResponse(brokerResp){};
    EnvInfo(BrokerResponse<T> brokerResp): brokerResponse(brokerResp){};
    virtual ~EnvInfo(){};
  };

  using EnvInfoSingle = EnvInfo<double>;
  using EnvInfoMulti = EnvInfo<PriceVector>;

  // struct SRDI{
  //   State state;
  //   double reward;
  //   bool done;
  //   std::unique_ptr<Info> info;
  // };

  // typedef std::tuple<State, double, bool, std::unique_ptr<Info>> SRDI;
  // typedef std::tuple<State, double, bool, EnvInfo<double>> SRDI;
  // typedef std::tuple<State, double, bool, EnvInfo<PriceVector>> SRDI;

  template<typename T>
  using SRDI = std::tuple<State, double, bool, EnvInfo<T>>;
  using SRDISingle = std::tuple<State, double, bool, EnvInfo<double>>;
  using SRDIMulti = std::tuple<State, double, bool, EnvInfo<PriceVector>>;


}


#endif // DATATYPES_H_
