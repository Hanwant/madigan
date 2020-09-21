#include "Account.h"

namespace madigan{

  void Account::addPortfolio(Portfolio port){
    if (portfolioBook_.find(port.id()) == portfolioBook_.end()){
      portfolios_.push_back(port);
      Portfolio* m_port = &(portfolios_[portfolios_.size()-1]);
      portfolioBook_[port.id()] = m_port;
      if (portfolios_.size() == 1){
        setDefaultPortfolio(m_port);
        // std::cout << "Setting default port: " << m_port->id() << "\n";
      }
      for(auto asset: m_port->assets()){
        auto found =std::find(assets_.begin(), assets_.end(), asset);
        if (found == assets_.end()){
          assets_.push_back(asset);
          defaultPrices_.push_back(0.);
        }
      }
      if (!registeredDataSource){
          new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
        }
      else{
        m_port->setDataSource(dataSource_);
      }
      cash_.conservativeResize(cash_.size()+1, Eigen::NoChange);
      borrowedCash_.conservativeResize(borrowedCash_.size()+1, Eigen::NoChange);
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

  void Account::setDefaultPortfolio(string accId){
    auto acc = portfolioBook_.find(accId);
    if (acc != portfolioBook_.end()){
      defaultPortfolio_ = acc->second;
    }
  }
  void Account::setDefaultPortfolio(Portfolio* portfolio){
    auto found = portfolioBook_.find(portfolio->id());
    if (found != portfolioBook_.end()){
      defaultPortfolio_= found->second;
    }
    else{
      addPortfolio(*portfolio);
      setDefaultPortfolio(portfolio->id());
    }
  }
  // Ledger Account::portfolio(){
  //   return portfolioBook_[];
  // }

  void Account::setDataSource(DataSource* source){
    dataSource_ = source;
    new (&currentPrices_) PriceVectorMap(source->currentData().data(), source->currentData().size());
    for (auto&& port: portfolios_){
      port.setDataSource(source);
    }
    registeredDataSource = true;
  }

  double Account::equity(){
    double sum{0};
    for (auto port: portfolios_){
      sum += port.portfolio_.dot(currentPrices_);
    }
    sum += cash_.sum();
    return sum;
  }

  double Account::availableMargin(){
    return cash(); //- borrowedMargin();
  }

}
