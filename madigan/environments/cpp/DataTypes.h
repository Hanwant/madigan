#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <memory>
#include <unordered_map>
#include <any>

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

  struct State{
    PriceVector price;
    Ledger portfolio;
    std::size_t timestamp;
  };

  struct DataItem{
    PriceVector prices;
    // TimeStamp timestamp;
  };

  struct Info{
    Info(){};
    virtual ~Info(){};
  };

  struct BrokerResponse: public Info{
  public:
    std::string event;
    PriceVector transactionPrices;
    PriceVector transactionCosts;

  public:
    BrokerResponse(){};
    BrokerResponse(PriceVector transPrices, PriceVector transCosts):
      event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
    BrokerResponse(std::string event, PriceVector transPrices, PriceVector transCosts):
      event(event), transactionPrices(transPrices), transactionCosts(transCosts){}
  };

  // struct SRDI{
  //   State state;
  //   double reward;
  //   bool done;
  //   std::unique_ptr<Info> info;
  // };

  typedef std::tuple<State, double, bool, std::unique_ptr<Info>> SRDI;






}


#endif // DATATYPES_H_
