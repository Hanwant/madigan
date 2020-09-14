#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include <string>
#include <memory>
#include <unordered_map>

#include "Portfolio.h"

namespace madigan{

  using std::string;
  // typedef std::unordered_map<std::string, std::shared_ptr<Portfolio>> PortfolioBook;
  typedef std::unordered_map<std::string, Portfolio> PortfolioBook;
  typedef std::unordered_map<std::string, std::vector<double>> AccountPortfolio;

  class Account{
  public:
    Account(){};
    Account(Portfolio &port) { addPortfolio(port);};
    Account(string id, Assets assets, double initCash): ID(id){ addPortfolio(id, assets, initCash);};
    Account(Assets assets, double initCash) { addPortfolio(assets, initCash);};
    ~Account(){};

    const string id(){ return ID;}

    // void addPortfolio(Portfolio &port);
    void addPortfolio(Portfolio &port);
    void addPortfolio(string id, Assets assets, double initCash);
    void addPortfolio(Assets assets, double initCash);

    AccountPortfolio portfolio();
    double cash();
    double eq();
    double availableMargin();
    double borrowedCash();
    friend class Broker;
  private:
    PortfolioBook Portfolios;
    string ID;
  };

} // namespace madigan

#endif // ACCOUNT_H_
