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

  struct BrokerResponse: public Info{
  public:
    std::string event;
    PriceVector transactionPrices;
    PriceVector transactionCosts;

  public:
    BrokerResponse(){};
    BrokerResponse(PriceVector transPrices, PriceVector transCosts):
      event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
    BrokerResponse(std::string event, PriceVector transPrices, PriceVector transCosts):
      event(event), transactionPrices(transPrices), transactionCosts(transCosts){}
  };

  class Broker{
  public:
    // Broker(){};
    Broker(const Account& account);
    Broker(const Portfolio& portfolio);
    Broker(Assets assets, double initCash);
    Broker(string AccId, Assets assets, double initCash);
    ~Broker(){};

    void addAccount(const Account& account);
    void setDefaultAccount(string accId);
    void setDefaultAccount(Account* account);
    void setSlippage(double slippagePct=0., double slippageAbs=0.);
    void setTransactionCosts(double transactionPct=0., double transactionAbs=0.);
    void setDataSource(DataSource* source);
    void setRequiredMargin(double reqMargin){
      defaultAccount_->setRequiredMargin(reqMargin); }
    void setRequiredMargin(string accID, double reqMargin){
      accountBook_.at(accID)->setRequiredMargin(reqMargin);}

    // const PriceVector& currentPrices(){ return dataSource_->currentData();}
    const Account& account() const { return *defaultAccount_;}
    const Account& account(string accID) const { return *(accountBook_.at(accID));}
    const Account& defaultAccount() const{ return *defaultAccount_;}
    const AccountBook& accountBook() const{return accountBook_;}
    std::unordered_map<string, Account> accountBookCopy() const;
    const std::vector<Account>& accounts() const{return accounts_;}
    const DataSource* dataSource() const{return dataSource_;}
    const PriceVectorMap currentPrices() const{return currentPrices_;}

    BrokerResponse handleEvent(AmountVector& units);
    BrokerResponse handleEvent(Order& order);
    BrokerResponse handleAction(AmountVector& units);
    BrokerResponse handleOrder(Order& order);

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

    bool checkRisk(double currencyAmount); // use default account & port
    bool checkRisk(int assetIdx, double units); // use default account & port
    bool checkRisk(string assetCode, double units); // use default account
    bool checkRisk(int assetIdx, double units, string accountID); // use specific account
    bool checkRisk(string assetCode, double units, string accountID); // use specific account

    double applySlippage(double price, double cash);
    double getTransactionCost(double cash);
    friend class Env;

  private:
    Assets assets_;
    std::unordered_map<string, unsigned int> assetIdx_;
    AccountBook accountBook_;
    std::vector<Account> accounts_;
    Account* defaultAccount_;
    Portfolio* defaultPortfolio_;

    std::vector<double> defaultPrices_;
    bool registeredDataSource{false};
    PriceVectorMap currentPrices_{nullptr, 0};

    DataSource* dataSource_{nullptr};

    double slippagePct_;
    double slippageAbs_;
    double transactionPct_;
    double transactionAbs_;
  };

}

#endif

