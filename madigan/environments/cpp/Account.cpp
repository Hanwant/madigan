#include "Account.h"

namespace madigan{

  void Account::addPortfolio(Portfolio &port){
    if (Portfolios.find(port.id()) == Portfolios.end()){
      // Portfolios[port.id()] = std::shared_ptr<Portfolio>(&port);
      Portfolios[port.id()] = port;
    }
    else{
      throw std::logic_error("Portfolio id must be unique and not already associated with Account");
    }
  }

  void Account::addPortfolio(Assets assets, double initCash){
    string id = "port_" + std::to_string(Portfolios.size());
    // Portfolios[id] = std::make_shared<Portfolio>(id, assets, initCash);
    Portfolios[id] = Portfolio(id, assets, initCash);
  }

  void Account::addPortfolio(string id, Assets assets, double initCash){
    if (Portfolios.find(id) == Portfolios.end()){
      // Portfolios[id] = std::make_shared<Portfolio>(id, assets, initCash);
      Portfolios[id] = Portfolio(id, assets, initCash);
    }
    else{
      throw std::logic_error("Portfolio id must be unique and not already associated with Account");
    }
  }

}
