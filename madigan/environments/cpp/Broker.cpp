#include "Broker.h"


namespace madigan {

  Broker::Broker(Account account){
    addAccount(account);
  }

  Broker::Broker(Portfolio portfolio){
    Account account(portfolio);
    addAccount(account);
  }

  Broker::Broker(Assets assets, double initCash){
    Account account(assets, initCash);
    addAccount(account);
  }

  Broker::Broker(string AccId, Assets assets, double initCash){
    Account account(AccId, assets, initCash);
    addAccount(account);
  }

  bool Broker::addAccount(Account account){
    if (accountBook_.find(account.id()) == accountBook_.end()){
      accountBook_[account.id()] = account;
      if(accountBook_.size() == 1){
        setDefaultAccount(account);
      }
      return true;
    }
    else{
      return false;
    }
  }

  void Broker::setDefaultAccount(string accId){
    auto acc = accountBook_.find(accId);
    if (acc != accountBook_.end()){
      defaultAccount_ = &(acc->second);
    }
  }

  void Broker::setDefaultAccount(Account &account){
    auto found = accountBook_.find(account.id());
    if (found != accountBook_.end()){
      defaultAccount_ = &(found->second);
    }
    else{
      addAccount(account);
      setDefaultAccount(account);
    }
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
        std::pair<double, double> transactionInfo;
        if(unit != 0. && checkRisk(assetIdx, unit)){
          transactionInfo = transaction(assetIdx, unit);
        }
        transPrices(assetIdx) = transactionInfo.first;
        transCosts(assetIdx) = transactionInfo.second;
      }
    }
    return BrokerResponse(transPrices, transCosts);
  }

  bool Broker::checkRisk(int assetIdx, double units){
    double currencyAmount=currentPrices_->operator()(assetIdx) * units;
    if (currencyAmount < defaultAccount_->availableMargin() - defaultAccount_->maintenanceMargin()){
      return true;
    }
    else return false;
  }

  std::pair<double, double> Broker::transaction(int assetIdx, double units){
    // Acounting for purchasing assets
    double transactionPrice = applySlippage(currentPrices_->operator()(assetIdx), units);
    double& currentUnits = defaultAccount_->defaultPortfolio_->portfolio_[assetIdx];
    currentUnits += units;
    // Accounting for adjusting cash
    double currencyAmount= transactionPrice*units;
    double transactionCost = abs(units)*transactionPct_ + transactionAbs_;
    defaultAccount_->defaultPortfolio_->cash_ -= (currencyAmount + transactionCost);
    return std::make_pair(transactionPrice, transactionCost);
  }

  double Broker::applySlippage(double price, double units){
    double slippage{(price*slippagePct_) + slippageAbs_};
    if (units<0){ // if selling
      return price - slippage;
    }
    else{ // if buying
      return price + slippage;
    }
  }

}
