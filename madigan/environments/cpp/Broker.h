#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>
#include <memory>
#include <iostream>

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
    // Broker(Account& account);
    // Broker(Portfolio& portfolio);
    Broker(Assets assets, double initCash);
    Broker(string AccId, Assets assets, double initCash);
    ~Broker(){};

    bool addAccount(Account& account);
    void setDefaultAccount(string accId);
    void setDefaultAccount(Account* account);
    void setSlippage(double slippagePct=0., double slippageAbs=0.);
    void setTransactionCosts(double transactionPct=0., double transactionAbs=0.);
    void setDataSource(DataSource* source);

    // const PriceVector& currentPrices(){ return dataSource_->currentData();}
    const Account& account() const { return *defaultAccount_;}
    const Account& defaultAccount() const{ return *defaultAccount_;}
    const AccountBook& accountBook() const{return accountBook_;}
    const std::vector<Account>& accounts() const{return accounts_;}
    const DataSource* dataSource() const{return dataSource_;}
    const PriceVectorMap currentPrices() const{return currentPrices_;}

    BrokerResponse handleEvent(AmountVector& units);
    BrokerResponse handleEvent(Order& order);

  private:
    BrokerResponse handleOrder(Order& order);
    BrokerResponse handleAction(AmountVector& units);

    bool checkRisk(double currencyAmount); // use default account
    bool checkRisk(int assetIdx, double units); // use default account
    bool checkRisk(string assetCode, double units); // use default account
    bool checkRisk(int assetIdx, double units, string accountID); // use specific account
    bool checkRisk(string assetCode, double units, string accountID); // use specific account
    std::pair<double, double> transaction(int assetIdx, double units); // use default account
    std::pair<double, double> transaction(int assetIdx, double units, double currencyAmount); // use default account
    std::pair<double, double> transaction(string assetCode, double units); // use default account
    std::pair<double, double> transaction(int assetIdx, double units, string accountID); // use specific account
    std::pair<double, double> transaction(string assetCode, double units, string accountID); // use specific account

    double applySlippage(double price, double units);
    friend class Env;

  private:
    Assets assets_;
    AccountBook accountBook_;
    std::vector<Account> accounts_;
    Account* defaultAccount_;

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

