#include "Broker.h"


namespace madigan {

  Broker::Broker(const Account& account){
    addAccount(account);
  }

  Broker::Broker(const Portfolio& portfolio){
    Account account(portfolio);
    addAccount(account);
  }

  Broker::Broker(Assets assets, double initCash){
    string id = "acc_" + std::to_string(accounts_.size());
    Account account(id, assets, initCash);
    addAccount(account);
  }

  Broker::Broker(string AccId, Assets assets, double initCash){
    Account account(AccId, assets, initCash);
    addAccount(account);
  }

  void Broker::addAccount(const Account& account){
    if (accountBook_.find(account.id()) == accountBook_.end()){
      accounts_.push_back(account);
      Account* m_account = &(accounts_[accounts_.size()-1]);
      for(auto& acc: accounts_){ // because vector has been resized
        accountBook_[acc.id()] = &acc;
      }
      // accountBook_[account.id()] = m_account;
      if(accounts_.size() == 1){
        setDefaultAccount(m_account);
      }
      else{
        setDefaultAccount(defaultAccID_); // vector has been resized - update memory location
      }
      for(auto asset: m_account->assets()){
        auto found =std::find(assets_.begin(), assets_.end(), asset);
        if (found == assets_.end()){
          assets_.push_back(asset);
          assetIdx_[asset.code] = assets_.size()-1;
          defaultPrices_.push_back(0.);
        }
      }
      if (!registeredDataSource){
        new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
      }
      else{
        m_account->setDataSource(dataSource_);
      }
    }
    else{
      throw std::logic_error((string)"acc with this id already exists!"+" ("+account.id()+")");
    }
  }

  void Broker::setDefaultAccount(string accID){
    Account* acc = accountBook_.at(accID);
    defaultAccount_ = acc;
    defaultAccID_ = accID;
    setDefaultPortfolio(acc->defaultPortfolio_);
  }


  void Broker::setDataSource(DataSourceTick* source){
    dataSource_ = source;
    new (&currentPrices_) PriceVectorMap(source->currentPrices().data(),
                                         source->currentPrices().size());
    for(auto&& acc: accounts_){
      acc.setDataSource(source);
    }
    registeredDataSource=true;
  }

  void Broker::setSlippage(double slippagePct, double slippageAbs){
    slippagePct_ = slippagePct;
    slippageAbs_ = slippageAbs;
  };
  void Broker::setTransactionCost(double transactionPct, double transactionAbs){
    transactionPct_ = transactionPct;
    transactionAbs_ = transactionAbs;
  };

  const Portfolio& Broker::portfolio(string portID) const{
    for (const auto& acc: accounts_){
      auto found = acc.portfolioBook().find(portID);
      if (found != acc.portfolioBook().end()){
        return *(found->second);
      }
    }
    throw std::out_of_range(string("Broker doesn't contain portfolio with id: ")+portID);
  }

  std::vector<Portfolio> Broker::portfolios() const{
    std::vector<Portfolio> ports;
    for(const auto& acc: accounts_){
      for (const auto& port: acc.portfolios()){
        ports.push_back(port);
      }
    }
    return ports;
  }
  std::unordered_map<string, Portfolio> Broker::portfolioBook() const{
    std::unordered_map<string, Portfolio> book;
    for (const auto& acc: accounts_){
      for (const auto& port: acc.portfolios()){
        book.emplace(port.id(), port);
      }
      return book;
    }
  }

  std::unordered_map<string, Account> Broker::accountBookCopy() const {
    std::unordered_map<string, Account> book;
    for (const auto& acc: accounts_){
      book.emplace(make_pair(acc.id(), acc));
    }
    return book;
  }

  BrokerResponseSingle Broker::handleTransaction(Portfolio* port,
                                                 int assetIdx, double units){
    if (units != 0.){
      RiskInfo risk = port->checkRisk(assetIdx, units);
      if(risk == RiskInfo::green){
        const double& currentPrice = currentPrices_(assetIdx);
        double transactionPrice = applySlippage(currentPrice ,units);
        double transactionCost = getTransactionCost(units*currentPrice);
        port->handleTransaction(assetIdx, transactionPrice, units,
                                transactionCost);
        return BrokerResponseSingle(transactionPrice, units, transactionCost, risk,
                                    (port->checkRisk()==RiskInfo::margin_call)? true: false);
      }
      return BrokerResponseSingle(0., 0., 0., risk,
                                  (port->checkRisk()==RiskInfo::margin_call)? true: false);
    }
    return BrokerResponseSingle(0., 0., 0., RiskInfo::green,
                                (port->checkRisk()==RiskInfo::margin_call)? true: false);
  }

  BrokerResponseMulti Broker::handleTransaction(Portfolio* port, const AmountVector& units){
    PriceVector transPrices(units.size());
    PriceVector transUnits(units.size());
    PriceVector transCosts(units.size());
    std::vector<RiskInfo> riskInfo(units.size());
    for (int i=0; i<units.size(); i++){
      auto brokerResp = handleTransaction(port, i, units[i]);
      transPrices[i] = brokerResp.transactionPrice;
      transUnits[i] = brokerResp.transactionUnits;
      transCosts[i] = brokerResp.transactionCost;
      riskInfo[i] = brokerResp.riskInfo;
    }
    return BrokerResponseMulti(transPrices, transUnits, transCosts, riskInfo,
                               (port->checkRisk()==RiskInfo::margin_call)? true: false);
  }

  BrokerResponseSingle Broker::close(int assetIdx){
    Portfolio* port = defaultPortfolio_;
    double units = -(port->ledger()(assetIdx));
    const double& currentPrice = currentPrices_(assetIdx);
    double transactionPrice = applySlippage(currentPrice, units);
    double transactionCost = getTransactionCost(currentPrice*units);
    port->close(assetIdx, transactionPrice, transactionCost);
    return BrokerResponseSingle(transactionPrice, units, transactionCost, RiskInfo::green, // closing is always green
                                (port->checkRisk()==RiskInfo::margin_call)? true: false);
  }

  double Broker::applySlippage(double price, double units){
    double slippage{(price*slippagePct_) + slippageAbs_};
    return units<0? (price-slippage): (price+slippage);
  }
  double Broker::getTransactionCost(double amount){
    return abs(amount)*transactionPct_ + transactionAbs_;

  }


}
