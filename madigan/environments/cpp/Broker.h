#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <stdexcept>

#include "DataTypes.h"
#include "Portfolio.h"
#include "Account.h"

namespace madigan{
  typedef std::unordered_map<std::string, Account*> AccountBook;

  struct Order{
    string accountID;
    string portfolioID;
    string orderID;
    string assetCode;
    double units;
    double submittedPrice;
    double executedPrice;

    std::size_t submittedTime;
    std::size_t receivedTime;
    std::size_t executedTime;
  };


  class Broker{
  public:
    // Broker(){};
    Broker(const Account& account);
    Broker(const Portfolio& portfolio);
    Broker(Assets assets, double initCash);
    Broker(string AccId, Assets assets, double initCash);
    Broker(const Broker&)=delete;
    Broker& operator=(const Broker&)=delete;
    ~Broker(){};

    int nAssets() const{ return assets_.size(); }
    const Assets& assets() const{ return assets_; }

    void addAccount(const Account& account);
    void addPortfolio(const Portfolio& port){
      defaultAccount_->addPortfolio(port);
    }
    void addPortfolio(string accID, const Portfolio& port){
      accountBook_.at(accID)->addPortfolio(port);
    }
    void setDefaultAccount(string accId); // Main logic
    void setDefaultAccount(Account *account){
      setDefaultAccount(account->id());
    }
    void setDefaultPortfolio(Portfolio* portfolio){
      defaultPortfolio_ = portfolio;
    }
    void setSlippage(double slippagePct=0., double slippageAbs=0.);
    void setTransactionCosts(double transactionPct=0., double transactionAbs=0.);
    void setDataSource(DataSource* source);
    void setRequiredMargin(double reqMargin){
      defaultAccount_->setRequiredMargin(reqMargin); }
    void setRequiredMargin(string accID, double reqMargin){
      accountBook_.at(accID)->setRequiredMargin(reqMargin);}
    void setRequiredMargin(string accID, string portID, double reqMargin){
      accountBook_.at(accID)->setRequiredMargin(portID, reqMargin);}
    void setMaintenanceMargin(double mainMargin){
      defaultAccount_->setMaintenanceMargin(mainMargin); }
    void setMaintenanceMargin(string accID, double mainMargin){
      accountBook_.at(accID)->setMaintenanceMargin(mainMargin);}
    void setMaintenanceMargin(string accID, string portID, double mainMargin){
      accountBook_.at(accID)->setMaintenanceMargin(portID, mainMargin);}

    // const PriceVector& currentPrices(){ return dataSource_->currentData();}
    const Account& account() const { return *defaultAccount_;}
    const Account& account(string accID) const { return *(accountBook_.at(accID));}
    const Account& defaultAccount() const{ return *defaultAccount_;}
    const AccountBook& accountBook() const{return accountBook_;}
    std::unordered_map<string, Account> accountBookCopy() const;
    const std::vector<Account>& accounts() const{return accounts_;}
    const DataSource* dataSource() const{return dataSource_;}
    const PriceVectorMap& currentPrices() const{return currentPrices_;}
    const Portfolio& portfolio() const { return *defaultPortfolio_; }
    const Portfolio& portfolio(string portID) const;
    vector<Portfolio> portfolios() const;
    const vector<Portfolio>& portfolios(string accID) const {
      return accountBook_.at(accID)->portfolios();}
    const PortfolioBook& portfolioBook(string accID) const {
      return accountBook_.at(accID)->portfolioBook(); }
    std::unordered_map<string, Portfolio> portfolioBook() const;

    BrokerResponseMulti handleEvent(const AmountVector& units){
      return handleTransaction(units);
    }

    BrokerResponseMulti handleAction(const AmountVector& units){
      return handleTransaction(units);
    }

    BrokerResponseSingle handleEvent(Order& order);
    BrokerResponseSingle handleOrder(Order& order);

    BrokerResponseSingle handleTransaction(int assetIdx, double units){
      return handleTransaction(defaultPortfolio_, assetIdx, units);
    }
    BrokerResponseSingle handleTransaction(string assetCode, double units){
      return handleTransaction(defaultPortfolio_, assetIdx_.at(assetCode), units);
    }
    BrokerResponseSingle handleTransaction(string accID, int assetIdx,
                                           double units){
      return handleTransaction(accountBook_.at(accID)->defaultPortfolio_,
                               assetIdx, units);
    }
    BrokerResponseSingle handleTransaction(string accID, string assetCode,
                                           double units){
      return handleTransaction(accountBook_.at(accID)->defaultPortfolio_,
                               assetIdx_.at(assetCode), units);
    }
    BrokerResponseSingle handleTransaction(string accID, string portID,
                                                   int assetIdx, double units){
      return handleTransaction(accountBook_.at(accID)->portfolioBook_.at(portID),
                               assetIdx, units);
    }
    BrokerResponseSingle handleTransaction(string accID, string portID,
                                                   string assetCode, double units){
      return handleTransaction(accountBook_.at(accID)->portfolioBook_.at(portID),
                               assetIdx_.at(assetCode), units);
    }

    BrokerResponseMulti handleTransaction(const AmountVector& units){
      return handleTransaction(defaultPortfolio_, units);
    }
    BrokerResponseMulti handleTransaction(string accID, const AmountVector& units){
      return handleTransaction(accountBook_.at(accID)->defaultPortfolio_, units);
    }
    BrokerResponseMulti handleTransaction(string accID, string portID,
                                                  const AmountVector& units){
      return handleTransaction(accountBook_.at(accID)->portfolioBook_.at(portID),
                               units);
    }

    RiskInfo checkRisk() { return defaultPortfolio_->checkRisk();}
    RiskInfo checkRisk(string accID){
      return accountBook_.at(accID)->defaultPortfolio_->checkRisk();
    }
    RiskInfo checkRisk(string accID, string portID){
      return accountBook_.at(accID)->portfolioBook_.at(portID)->checkRisk();
    }
    RiskInfo checkRisk(int assetIdx, double units){// use default  port
      return defaultPortfolio_->checkRisk(assetIdx, units);
    }
    RiskInfo checkRisk(string assetCode, double units){// use default  port
      return defaultPortfolio_->checkRisk(assetCode, units);
    }
    RiskInfo checkRisk(string accID, int assetIdx, double units){ // use specific account
      return accountBook_.at(accID)->defaultPortfolio_->checkRisk(assetIdx, units);
    }
    RiskInfo checkRisk(string accID, string assetCode, double units){ // use specific account
      return accountBook_.at(accID)->defaultPortfolio_->checkRisk(assetCode, units);
    }
    RiskInfo checkRisk(string accID, string portID, int assetIdx, double units){ // use specific account and port
      return accountBook_.at(accID)->portfolioBook_.at(portID)->checkRisk(assetIdx, units);
    }
    RiskInfo checkRisk(string accID, string portID, string assetCode, double units){ // use specific account and port
      return accountBook_.at(accID)->portfolioBook_.at(portID)->checkRisk(assetCode, units);
    }

  private:
    // all public handleTransaction functions call this
    BrokerResponseSingle handleTransaction(Portfolio* port, int assetIdx, double units);
    BrokerResponseMulti handleTransaction(Portfolio* port, const AmountVector& units);

    // public checkRisk functions call one of these
    RiskInfo checkRisk(Portfolio* port){ return port->checkRisk(); }
    RiskInfo checkRisk(Portfolio* port, int assetIdx, double units){
      return port->checkRisk(assetIdx, units);}
    RiskInfo checkRisk(Portfolio* port, string assetCode, double units){
      return port->checkRisk(assetCode, units);}

    double applySlippage(double price, double cash);
    double getTransactionCost(double cash);
    friend class Env;

  private:
    Assets assets_;
    std::unordered_map<string, unsigned int> assetIdx_;
    AccountBook accountBook_;
    std::vector<Account> accounts_;
    Account* defaultAccount_{nullptr};
    string defaultAccID_;
    Portfolio* defaultPortfolio_{nullptr};

    std::vector<double> defaultPrices_;
    bool registeredDataSource{false};
    PriceVectorMap currentPrices_{nullptr, 0};

    DataSource* dataSource_{nullptr};

    double slippagePct_{0.};
    double slippageAbs_{0.};
    double transactionPct_{0.};
    double transactionAbs_{0.};
  };

}

#endif

