#include "Account.h"

namespace madigan{

  void Account::addPortfolio(Portfolio port){
    std::cout << "acc id: " << id() <<  " adding port: " << port << "\n";
    if (portfolios_.find(port.id()) == portfolios_.end()){
      portfolios_[port.id()] = port;
      setDefaultPortfolio(port);
      for(auto asset: port.assets()){
        auto found =std::find(assets_.begin(), assets_.end(), asset);
        if (found == assets_.end()){
          assets_.push_back(asset);
        }
      }
      cash_.conservativeResize(cash_.size()+1, Eigen::NoChange);
      borrowedCash_.conservativeResize(borrowedCash_.size()+1, Eigen::NoChange);
      // port.cash_ = cash_(portfolios_.size()-1);
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

  // PortfolioBook Account::portfolios(){
  //   for(auto iter=portfolios_.begin(); iter!=portfolios_.end(); iter++){
  //     std::cout<< "key: " << iter->first << "\n";
  //     std::cout<< "val: " << iter->second<< "\n";
  //   }
  //   return portfolios_;
  // }
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


  double Account::equity(){
    double sum{0};
    for (auto port: portfolios_){
      sum += port.second.portfolio_.dot(*currentPrices_);
    }
    sum += cash_.sum();
    return sum;
  }

  double Account::availableMargin(){
    return cash() - borrowedMargin();
  }

}
