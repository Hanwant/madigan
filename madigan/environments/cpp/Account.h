#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include <string>
#include <memory>
#include <unordered_map>
#include <numeric>

#include <pybind11/pybind11.h>

#include "DataSource.h"
#include "Portfolio.h"

namespace madigan{

  using std::string;
  typedef std::unordered_map<std::string, Portfolio*> PortfolioBook;
  typedef std::unordered_map<std::string, vector<double>> AccountPortfolio;

  class Account{
  public:
    // Account(){};
    // Account(): id_("account_default"){ addPortfolio(Portfolio());};
    Account(const Portfolio& port): id_("acc_"+port.id()) { addPortfolio(port);};
    Account(string id, Assets assets, double initCash): id_(id){ addPortfolio(assets, initCash);};
    Account(Assets assets, double initCash) { addPortfolio(assets, initCash);};
    Account(const Account& other);
    Account& operator=(const Account& other);
    Account(const Account&& other)=delete;
    Account& operator=(const Account&& other)=delete;
    ~Account(){};

    const string id() const{ return id_;}

    // void addPortfolio(Portfolio &port);
    void addPortfolio(const Portfolio& port);
    void addPortfolio(pybind11::object pyport);
    void addPortfolio(string id, Assets assets, double initCash);
    void addPortfolio(Assets assets, double initCash);
    void setDataSource(DataSource* source);
    void setRequiredMargin(double reqMargin){ requiredMargin_=reqMargin;}

    const DataSource* dataSource() const {return dataSource_;}
    const PriceVectorMap& currentPrices() const{ return currentPrices_;}

    Assets assets(){ return assets_;}
    // Portfolio& portfolio() const { return *defaultPortfolio_;}
    // Portfolio& defaultPortfolio() const {return *defaultPortfolio_;}
    const Portfolio& portfolio() const { return *defaultPortfolio_;}
    const Portfolio& portfolio(string portID) const
    { return *(portfolioBook_.at(portID));}
    const Portfolio& defaultPortfolio() const {return *defaultPortfolio_;}
    const PortfolioBook& portfolioBook() const {return portfolioBook_;}
    std::unordered_map<string, Portfolio> portfolioBookCopy() const;
    const std::vector<Portfolio>& portfolios() const {return portfolios_;}

    void setDefaultPortfolio(string portId);
    void setDefaultPortfolio(Portfolio* portfolio);

    int nAssets() const { return assets_.size(); }
    Assets assets() const { return assets_; }
    double initCash() const;
    double initCash(string portID) const;
    double cash() const;
    double cash(string portID) const;
    Ledger ledger() const;
    const Ledger& ledger(string portID) const;
    Ledger ledgerNormed() const;
    Ledger ledgerNormed(string portID) const;
    /* double usedMargin() const { return usedmargin.sum()} */
    double assetValue() const;
    double assetValue(string portID) const;
    double equity() const;
    double equity(string portID) const;
    double availableMargin() const;
    double availableMargin(string portID) const;
    double requiredMargin() const{return requiredMargin_;} // account wide
    // double requiredMargin(string portID) const;
    double borrowedMargin() const;
    double borrowedMargin(string portID) const;
    double borrowedCash() const;

    void handleTransaction(string assetCode, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);
    void handleTransaction(int assetIdx, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);

    void handleTransaction(string portId, string assetCode, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);
    void handleTransaction(string portId, int assetIdx, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);

    friend class Broker;
    friend class Env;
  private:
    std::vector<Portfolio> portfolios_;
    PortfolioBook portfolioBook_;
    string id_{"acc_default"};
    Portfolio* defaultPortfolio_{nullptr};
    string defaultPortID_;
    Assets assets_;
    int nAssets_;
    double maintenanceMargin_{0.25};
    double requiredMargin_{1.}; // default = no levarage need 100% margin
    double borrowedMarginRatio_{1./requiredMargin_ -1.}; // defaults to 0.
    Ledger cash_;
    // Ledger usedMargin_;
    double borrowedMargin_;
    double balance_;
    Ledger borrowedCash_;

    std::vector<double> defaultPrices_;
    bool registeredDataSource{false};
    PriceVectorMap currentPrices_{nullptr, 0};
    DataSource* dataSource_{nullptr};

  };


} // namespace madigan

#endif // ACCOUNT_H_
