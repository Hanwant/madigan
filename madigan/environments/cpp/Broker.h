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

    void addAccount(const Account& account);
    void addPortfolio(const Portfolio& port);
    void addPortfolio(string accID, const Portfolio& port);
    void setDefaultAccount(string accId);
    void setDefaultAccount(Account*);
    void setDefaultPortfolio(Portfolio*);
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

    BrokerResponseMulti handleEvent(AmountVector& units);
    BrokerResponseSingle handleEvent(Order& order);
    BrokerResponseMulti handleAction(AmountVector& units);
    BrokerResponseSingle handleOrder(Order& order);

    std::pair<double, double> handleTransaction(int assetIdx, double units); // use default account
    std::pair<double, double> handleTransaction(string assetCode, double units); // use default account
    std::pair<double, double> handleTransaction(string accountID, int assetIdx,
                                                double units); // use specific account
    std::pair<double, double> handleTransaction(string accountID, string assetCode,
                                                double units); // use specific account
    std::pair<double, double> handleTransaction(string accountID, string portID,
                                                int assetIdx, double units); // use specific account & port
    std::pair<double, double> handleTransaction(string accountID, string portID,
                                                string assetCode, double units); // use specific account & port

  private:
    // all public handleTransaction functions call this
    std::pair<double, double> handleTransaction(Account* acc, Portfolio* port,
                                                int assetIdx, double units);
    // std::pair<double, double> handleTransaction(Account* acc, Portfolio* port,
    //                                             string assetCode, double units);

    // bool checkRisk(double currencyAmount); // use default account & port
    RiskInfo checkRisk();
    RiskInfo checkRisk(int assetIdx, double units); // use default account & port
    RiskInfo checkRisk(string assetCode, double units); // use default account
    RiskInfo checkRisk(int assetIdx, double units, string accountID); // use specific account
    RiskInfo checkRisk(string assetCode, double units, string accountID); // use specific account

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

