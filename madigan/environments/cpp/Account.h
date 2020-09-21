#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include <string>
#include <memory>
#include <unordered_map>

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
    Account(Portfolio& port): id_("acc_"+port.id()) { addPortfolio(port);};
    Account(string id, Assets assets, double initCash): id_(id){ addPortfolio(assets, initCash);};
    Account(Assets assets, double initCash) { addPortfolio(assets, initCash);};
    ~Account(){};

    const string id(){ return id_;}

    // void addPortfolio(Portfolio &port);
    void addPortfolio(Portfolio port);
    void addPortfolio(string id, Assets assets, double initCash);
    void addPortfolio(Assets assets, double initCash);
    void setDataSource(DataSource* source);

    const DataSource* dataSource() const {return dataSource_;}
    const PriceVectorMap& currentPrices() const{ return currentPrices_;}

    Assets assets(){ return assets_;}
    // Portfolio& portfolio() const { return *defaultPortfolio_;}
    // Portfolio& defaultPortfolio() const {return *defaultPortfolio_;}
    const Portfolio& portfolio() const { return *defaultPortfolio_;}
    const Portfolio& defaultPortfolio() const {return *defaultPortfolio_;}
    PortfolioBook portfolioBook() const {return portfolioBook_;}
    const std::vector<Portfolio>& portfolios() const {return portfolios_;}

    double cash() { return cash_.sum();}
    // double currentValue() { return dataSource_->currentData()->dot(defaultPortfolio_->portfolio_);}
    // double borrowedCash() {return borrowedCash_.sum();}
    // double borrowedMargin() {return usedMargin_ * borrowedMarginRatio_}
    double equity();
    double availableMargin();
    double requiredMargin() { return requiredMargin_;}
    double maintenanceMargin(){return equity()*maintenanceMargin_;}

    friend class Broker;
    friend class Env;
  private:
    void setDefaultPortfolio(string portId);
    void setDefaultPortfolio(Portfolio* portfolio);

  private:
    PortfolioBook portfolioBook_;
    std::vector<Portfolio> portfolios_;
    string id_{"acc_default"};
    Portfolio* defaultPortfolio_;
    Assets assets_;
    double maintenanceMargin_{0.25};
    double requiredMargin_{1.}; // default = no levarage need 100% margin
    double borrowedmarginRatio_{1./requiredMargin_ -1.}; // defaults to 0.
    Ledger cash_;
    Ledger usedMargin_;
    double balance_;
    Ledger borrowedCash_;

    std::vector<double> defaultPrices_;
    bool registeredDataSource{false};
    PriceVectorMap currentPrices_{nullptr, 0};
    DataSource* dataSource_{nullptr};

  };


} // namespace madigan

#endif // ACCOUNT_H_
