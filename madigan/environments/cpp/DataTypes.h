#ifndef DATAYPES_H_
#define DATAYPES_H_

#include <memory>

#include <Eigen/Core>

namespace madigan{

  // template<typename T>
  // struct Data{
  //   Data();
  //   ~Data();
  //   T data;
  // };

  typedef Eigen::VectorXi ActionVector;
  typedef Eigen::VectorXd PriceVector;
  typedef Eigen::Map<const PriceVector> PriceVectorMap;
  typedef Eigen::VectorXd AmountVector;
  typedef Eigen::VectorXd Ledger;

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
