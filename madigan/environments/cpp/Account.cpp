#include "Account.h"

namespace madigan{

  void Account::addPortfolio(Portfolio port){
    std::cout << "acc id: " << id() <<  " adding port: " << port << "\n";
    if (portfolios_.find(port.id()) == portfolios_.end()){
      // portfolios_[port.id()] = std::shared_ptr<Portfolio>(&port);
      portfolios_[port.id()] = port;
      setDefaultPortfolio(port);
    }
    else{
      throw std::logic_error("Portfolio id must be unique and not already associated with Account");
    }
  }

  void Account::addPortfolio(string id, Assets assets, double initCash){
    Portfolio port=Portfolio(id, assets, initCash);
    addPortfolio(port);
  }

  void Account::addPortfolio(Assets assets, double initCash){
    string id = "port_" + std::to_string(portfolios_.size());
    addPortfolio(id, assets, initCash);
  }

  PortfolioBook Account::portfolios(){
    for(auto iter=portfolios_.begin(); iter!=portfolios_.end(); iter++){
      std::cout<< "key: " << iter->first << "\n";
      std::cout<< "val: " << iter->second<< "\n";
    }
    return portfolios_;
  }
  void Account::setDefaultPortfolio(string accId){
    auto acc = portfolios_.find(accId);
    if (acc != portfolios_.end()){
      defaultPortfolio_ = &(acc->second);
    }
  }
  void Account::setDefaultPortfolio(Portfolio &portfolio){
    auto found = portfolios_.find(portfolio.id());
    if (found != portfolios_.end()){
      defaultPortfolio_= &(found->second);
    }
    else{
      addPortfolio(portfolio);
      setDefaultPortfolio(portfolio);
    }
  }
  // Ledger Account::portfolio(){
  //   return portfolios_[];
  // }

}
