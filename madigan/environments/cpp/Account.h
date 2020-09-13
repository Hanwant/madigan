#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include <string>
#include <memory>
#include <unordered_map>

#include "Portfolio.h"

namespace madigan{

  using std::string;
  typedef std::unordered_map<std::string, std::unique_ptr<Portfolio>> PortfolioBook;

  class Account{
  public:
    Account();
    Account(Portfolio *port){ addPortfolio(port);};
    Account(string id, int nAssets, double initCash){ addPortfolio(id, nAssets, initCash);};
    Account(int nAssets, double initCash){ addPortfolio(nAssets, initCash);};
    ~Account(){};
    void addPortfolio(Portfolio *port);
    void addPortfolio(string id, int nAssets, double initCash);
    void addPortfolio(int nAssets, double initCash);
    friend class Broker;
  private:
    PortfolioBook Portfolios;
    const string ID;
  };

} // namespace madigan

#endif // ACCOUNT_H_
