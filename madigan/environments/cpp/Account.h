#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include <string>
#include <memory>
#include <unordered_map>

#include "Portfolio.h"

namespace madigan{

  using std::string;
  typedef std::unordered_map<std::string, Portfolio> PortfolioBook;
  typedef std::unordered_map<std::string, vector<double>> AccountPortfolio;
  // std::ostream& operator<<(std::ostream& os, const PortfolioBook& book){
  //   for(auto it=book.begin(); it!=book.end(); it ++){
  //     os << it->second << "\n";
  //   }
  //   return os;
  // }

  class Account{
  public:
    Account(): id_("account_default"){ addPortfolio(Portfolio());};
    Account(Portfolio port): id_(port.id()) { addPortfolio(port);};
    Account(string id, Assets assets, double initCash): id_(id){ addPortfolio(assets, initCash);};
    Account(Assets assets, double initCash) { addPortfolio(assets, initCash);};
    ~Account(){};

    const string id(){ return id_;}

    // void addPortfolio(Portfolio &port);
    void addPortfolio(Portfolio port);
    void addPortfolio(string id, Assets assets, double initCash);
    void addPortfolio(Assets assets, double initCash);

    Portfolio portfolio(){ return *defaultPortfolio_;};
    Portfolio defaultPortfolio(){return *defaultPortfolio_;};
    PortfolioBook portfolios();
    double cash();
    double eq();
    double availableMargin();
    double borrowedCash();
    friend class Broker;
  private:
    void setDefaultPortfolio(string portId);
    void setDefaultPortfolio(Portfolio& portfolio);
  private:
    PortfolioBook portfolios_;
    string id_;
    Portfolio* defaultPortfolio_;
  };

} // namespace madigan

#endif // ACCOUNT_H_
