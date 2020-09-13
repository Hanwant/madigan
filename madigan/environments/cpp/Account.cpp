#include "Account.h"

namespace madigan{

  void Account::addPortfolio(Portfolio *port){
    if (Portfolios.find(port->id()) == Portfolios.end()){
      Portfolios[port->id()] = std::unique_ptr<Portfolio>(port);
    }
    else{
      throw std::logic_error("Portfolio id must be unique and not already associated with Account");
    }
  }

  void Account::addPortfolio(int nAssets, double initCash){
    string id = "port_" + std::to_string(Portfolios.size());
    Portfolios[id] = std::make_unique<Portfolio>(id, nAssets, initCash);
  }

  void Account::addPortfolio(string id, int nAssets, double initCash){
    Portfolios[id] = std::make_unique<Portfolio>(id, nAssets, initCash);
  }

}
