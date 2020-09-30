#include "Account.h"

namespace madigan{

  Account::Account(const Account& other){
    // std::cout << "ACC COPY CONS BEGIN" << "\n";
    id_ = other.id_;
    portfolios_ = other.portfolios_;
    for (auto& port: portfolios_){
      portfolioBook_[port.id()] = &port;
    }
    defaultPortID_ = other.defaultPortID_;
    defaultPortfolio_ = portfolioBook_[defaultPortID_];
    assets_=other.assets_;
    nAssets_ = assets_.size();
    // maintenanceMargin_ = other.maintenanceMargin_;
    // requiredMargin_ = other.requiredMargin_;
    // borrowedMargin_ = other.borrowedMargin_;
    // borrowedMarginRatio_ = other.borrowedMarginRatio_;
    cash_ = other.cash_;
    balance_ = other.balance_;
    borrowedCash_ = other.borrowedCash_;
    defaultPrices_ = other.defaultPrices_;
    registeredDataSource = other.registeredDataSource;
    dataSource_ = other.dataSource_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           other.currentPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(),
                                           defaultPrices_.size());
    }
  }

  Account& Account::operator=(const Account& other){
    // std::cout << "ACC COPY ASSIGN BEGIN " << "\n";
    id_ = other.id_;
    portfolios_ = other.portfolios_;
    for (auto& port: portfolios_){
      portfolioBook_[port.id()] = &port;
    }
    defaultPortID_ = other.defaultPortID_;
    defaultPortfolio_ = portfolioBook_[defaultPortID_];
    assets_=other.assets_;
    nAssets_=assets_.size();
    // maintenanceMargin_=other.maintenanceMargin_;
    // requiredMargin_=other.requiredMargin_;
    // borrowedMargin_ = other.borrowedMargin_;
    // borrowedMarginRatio_ = other.borrowedMarginRatio_;
    cash_ = other.cash_;
    balance_ = other.balance_;
    borrowedCash_ = other.borrowedCash_;
    defaultPrices_ = other.defaultPrices_;
    registeredDataSource = other.registeredDataSource;
    dataSource_ = other.dataSource_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           other.currentPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(),
                                           defaultPrices_.size());
    }
    return *this;
  }

  void Account::addPortfolio(const Portfolio& port){
    if (defaultPortfolio_){
      if (port.assets().size() != assets_.size()){
        std::cout<< port.assets().size() << ", " << assets_.size() << "\n";
        throw std::length_error("portfolio must have the same number of assets as account");
      }
    }
    if (portfolioBook_.find(port.id()) == portfolioBook_.end()){
      portfolios_.push_back(port);
      Portfolio* m_port = &(portfolios_[portfolios_.size()-1]);
      for(auto& port: portfolios_){
        portfolioBook_[port.id()] = &port;
      }
      if (portfolios_.size() == 1){
        setDefaultPortfolio(m_port);
      }
      else{
        setDefaultPortfolio(defaultPortID_); // BECAUSE VECTOR HAS BEEN RESIZED - UPDATE MEMORY LOCATION
      }
      for(auto asset: m_port->assets()){
        auto found = std::find(assets_.begin(), assets_.end(), asset);
        if (found == assets_.end()){
          assets_.push_back(asset);
          defaultPrices_.push_back(0.);
          nAssets_ += 1;
        }
      }
      if (!registeredDataSource){
          new (&currentPrices_) PriceVectorMap(defaultPrices_.data(),
                                               defaultPrices_.size());
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

  void Account::addPortfolio(pybind11::object pyport){
    Portfolio* port = pyport.cast<Portfolio*>();
    addPortfolio(*port);
  }

  void Account::addPortfolio(string id, Assets assets, double initCash){
    Portfolio port=Portfolio(id, assets, initCash);
    addPortfolio(port);
  }

  void Account::addPortfolio(Assets assets, double initCash){
    string id = "port_" + std::to_string(portfolios_.size());
    addPortfolio(id, assets, initCash);
  }

  void Account::setDefaultPortfolio(string portID){
    auto port = portfolioBook_.find(portID);
    if (port != portfolioBook_.end()){
      defaultPortfolio_ = port->second;
      defaultPortID_ = port->first;
    }
  }
  void Account::setDefaultPortfolio(Portfolio* portfolio){
    auto found = portfolioBook_.find(portfolio->id());
    if (found != portfolioBook_.end()){
      defaultPortfolio_= found->second;
      defaultPortID_ = found->first;
    }
    else{
      addPortfolio(*portfolio);
      setDefaultPortfolio(portfolio->id());
    }
  }
  void Account::setRequiredMargin(double reqMargin){
    defaultPortfolio_->setRequiredMargin(reqMargin);
  }
  void Account::setRequiredMargin(string portID, double reqMargin){
    portfolioBook_.at(portID)->setRequiredMargin(reqMargin);
  }
  void Account::setMaintenanceMargin(double mainMargin){
    defaultPortfolio_->setMaintenanceMargin(mainMargin);
  }
  void Account::setMaintenanceMargin(string portID, double mainMargin){
    portfolioBook_.at(portID)->setMaintenanceMargin(mainMargin);
  }

  std::unordered_map<string, Portfolio> Account::portfolioBookCopy() const{
    std::unordered_map<string, Portfolio> book;
    for (auto& port: portfolios_){
      book.emplace(std::make_pair(port.id(),  port));
    }
    return book;
  }
  // }
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

  double Account::initCash() const{
    // double sum{0.};
    // for (const auto& port: portfolios_){
    //   sum += port.initCash();
    // }
    // return sum;
    return std::accumulate(portfolios_.begin(), portfolios_.end(), 0.,
                           [] (double sum, const Portfolio& port){ return sum+(port.initCash()); });
  }
  double Account::initCash(string portID) const{
    return portfolioBook_.at(portID)->initCash();
  }

  double Account::cash() const{
    double sum{0.};
    for (const auto& port: portfolios_){
      sum += port.cash();
    }
    return sum;
  }
  double Account::cash(string portID) const{
    return portfolioBook_.at(portID)->cash();
  }

  Ledger Account::ledger() const{
    Ledger portfolioSum = Ledger::Zero(nAssets_);
    for (const auto& port: portfolios_){
      portfolioSum += port.ledger();
    }
    return portfolioSum;
  }
  const Ledger& Account::ledger(string portID) const{
    return portfolioBook_.at(portID)->ledger();
  }

  Ledger Account::ledgerNormed() const{
    Ledger portfolioNorm = Ledger::Zero(nAssets_);
    int i=0;
    for (const auto& port: portfolios_){
      portfolioNorm += port.ledgerNormed();
      i+=1;
    }
    return portfolioNorm.array() / i;
  }
  Ledger Account::ledgerNormed(string portID) const{
    return portfolioBook_.at(portID)->ledgerNormed();
  }

  /* double usedMargin() const { return usedmargin.sum()} */
  double Account::assetValue() const{
    double sum{0.};
    for (const auto& port: portfolios_){
      sum += port.assetValue();
    }
    return sum;
  }
  double Account::assetValue(string portID) const{
    return portfolioBook_.at(portID)->assetValue();
  }

  double Account::equity() const{
    double sum{0.};
    for(const auto& port: portfolios_){
      sum += port.equity();
    }
    return sum;
  }

  double Account::equity(string portID) const{
    return portfolioBook_.at(portID)->equity();
  }
  double Account::availableMargin() const{
    double sum{0.};
    for(const auto& port: portfolios_){
      sum += port.availableMargin();
    }
    return sum;
  }
  double Account::availableMargin(string portID) const{
    return portfolioBook_.at(portID)->availableMargin();
  }

  double Account::borrowedMargin() const{
    double sum{0.};
    for(const auto& port: portfolios_){
      sum += port.borrowedMargin();
    }
    return sum;
  }
  double Account::borrowedMargin(string portID) const{
    return portfolioBook_.at(portID)->borrowedMargin();
  }

  void Account::handleTransaction(string assetCode, double transactionPrice,
                         double units, double transactionCost){
    handleTransaction(defaultPortID_, assetCode, transactionPrice, units,
                      transactionCost);
  }
  void Account::handleTransaction(int assetIdx, double transactionPrice,
                         double units, double transactionCost){
    handleTransaction(defaultPortID_, assetIdx, transactionPrice, units,
                      transactionCost);
  }

  void Account::handleTransaction(string portId, string assetCode, double transactionPrice,
                                  double units, double transactionCost){
    Portfolio* port = portfolioBook_.at(portId);
    port->handleTransaction(assetCode, transactionPrice, units,
                            transactionCost);
  }
  void Account::handleTransaction(string portId, int assetIdx, double transactionPrice,
                         double units, double transactionCost){
    Portfolio* port = portfolioBook_.at(portId);
    port->handleTransaction(assetIdx, transactionPrice, units,
                            transactionCost);
  }
}
