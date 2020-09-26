#include "Broker.h"


namespace madigan {

  Broker::Broker(const Account& account){
    addAccount(account);
  }

  Broker::Broker(const Portfolio& portfolio){
    Account account(portfolio);
    addAccount(account);
  }

  std::unordered_map<string, Account> Broker::accountBookCopy() const {
    std::unordered_map<string, Account> book;
    for (const auto& acc: accounts_){
      book.emplace(make_pair(acc.id(), acc));
    }
    return book;
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
    defaultPortfolio_ = acc->defaultPortfolio_;
  }

  void Broker::setDefaultAccount(Account *account){
    setDefaultAccount(account->id());
  }

  void Broker::setDataSource(DataSource* source){
    dataSource_ = source;
    new (&currentPrices_) PriceVectorMap(source->currentData().data(), source->currentData().size());
    for(auto&& acc: accounts_){
      acc.setDataSource(source);
    }
    registeredDataSource=true;
  }

  void Broker::setSlippage(double slippagePct, double slippageAbs){
    slippagePct_ = slippagePct;
    slippageAbs_ = slippageAbs;
  };
  void Broker::setTransactionCosts(double transactionPct, double transactionAbs){
    transactionPct_ = transactionPct;
    transactionAbs_ = transactionAbs;
  };

  BrokerResponse Broker::handleEvent(AmountVector& units){
    return handleAction(units);
  }

  BrokerResponse Broker::handleAction(AmountVector& units){
    PriceVector transPrices(units.size());
    PriceVector transCosts(units.size());
    // std::vector<>
    if (units.size() == defaultAccount_->assets_.size()){
      for (int assetIdx=0; assetIdx<units.size(); assetIdx++){
        double unit = units[assetIdx];
        double transPrice{0.};
        double transCost{0.};
        if(unit != 0. && checkRisk(assetIdx, unit)){
          auto [transPrice, transCost] = handleTransaction(assetIdx, unit);
        }
        transPrices(assetIdx) = transPrice;
        transCosts(assetIdx) = transCost;
      }
    }
    return BrokerResponse(transPrices, transCosts);
  }

  bool Broker::checkRisk(int assetIdx, double units){
    // double currencyAmount=currentPrices_.operator()(assetIdx) * units;
    // double currencyAmount=currentPrices_(assetIdx) * units;
    // if (currencyAmount < defaultAccount_->availableMargin() - defaultAccount_->maintenanceMargin()){
    //   return true;
    // }
    // else return false;
    return true;
  }

  // Private version called by all public handleTransaction overloads
  std::pair<double, double> Broker::handleTransaction(Account* acc, Portfolio* port,
                                                      int assetIdx, double units){
    double transactionPrice = applySlippage(currentPrices_(assetIdx) ,units);
    double transactionCost = getTransactionCost(units);
    port->handleTransaction(assetIdx, transactionPrice, units,
                            transactionCost, acc->requiredMargin());
    return std::make_pair(transactionPrice, transactionCost);
  }
  std::pair<double, double> Broker::handleTransaction(int assetIdx, double units){
    Account* acc = defaultAccount_;
    Portfolio* port = acc->defaultPortfolio_;
    return handleTransaction(acc, port, assetIdx, units);
  }
  std::pair<double, double> Broker::handleTransaction(string assetCode, double units){
    Account* acc = defaultAccount_;
    Portfolio* port = acc->defaultPortfolio_;
    unsigned int assetIdx = assetIdx_.at(assetCode);
    return handleTransaction(acc, port, assetIdx, units);
  }
  std::pair<double, double> Broker::handleTransaction(string accID, int assetIdx,
                                                      double units){
    Account* acc = accountBook_.at(accID);
    Portfolio* port = acc->defaultPortfolio_;
    return handleTransaction(acc, port, assetIdx, units);
  }
  std::pair<double, double> Broker::handleTransaction(string accID, string assetCode,
                                                      double units){
    Account* acc = accountBook_.at(accID);
    Portfolio* port = acc->defaultPortfolio_;
    unsigned int assetIdx = assetIdx_.at(assetCode);
    return handleTransaction(acc, port, assetIdx, units);
  }
  std::pair<double, double> Broker::handleTransaction(string accID, string portID,
                                                      int assetIdx, double units){
    Account* acc = accountBook_.at(accID);
    Portfolio* port = acc->portfolioBook_.at(portID);
    return handleTransaction(acc, port, assetIdx, units);
  }
  std::pair<double, double> Broker::handleTransaction(string accID, string portID,
                                                      string assetCode, double units){
    Account* acc = accountBook_.at(accID);
    Portfolio* port = acc->portfolioBook_.at(portID);
    unsigned int assetIdx = assetIdx_.at(assetCode);
    return handleTransaction(acc, port, assetIdx, units);
  }

  double Broker::applySlippage(double price, double units){
    double slippage{(price*slippagePct_) + slippageAbs_};
    return units<0? price-slippage: price+slippage;
  }
  double Broker::getTransactionCost(double units){
    return abs(units)*transactionPct_ + transactionAbs_;

  }
}
