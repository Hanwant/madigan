#ifndef PORTFOLIO_H_
#define PORTFOLIO_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include<Eigen/Core>

#include "Assets.h"


namespace madigan{

  using Ledger=Eigen::VectorXd;
  using std::string;

  struct Info{
    std::string event;
    std::vector<double> transactionPrices;
    std::vector<double> transactionCosts;

  Info(std::string event, std::vector<double> transPrices, std::vector<double> transCosts):
    event(event), transactionPrices(transPrices), transactionCosts(transCosts){}

  Info(std::vector<double> transPrices, std::vector<double> transCosts):
    event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
    Info(){};

  };


  class Portfolio {
  public:
    Portfolio(){};
    Portfolio(Assets assets, double initCash);
    Portfolio(string id, Assets assets, double initCash);
    Portfolio(string id, Assets assets, double initCash, Ledger portfolio);
    Portfolio(string id, std::vector<string> assets, double initCash);
    Portfolio(string id, std::vector<string> assets, double initCash, Ledger portfolio);
    ~Portfolio()=default;
    string id() const { return id_;}
    Assets assets() const {return assets_;}
    std::unordered_map<string, unsigned int> assetIdx() const;
    unsigned int assetIdx(const string code) const;
    int nAssets() const { return assets_.size();}
    double initCash() const {return initCash_;}
    double cash() const { return cash_;}
    Ledger portfolio() const {return portfolio_;}
    Ledger portfolioNormed() const;
    double equity() const;
    double availableMargin() const;
    double borrowedCash() const;

    double operator[](string code) {
      return portfolio_[assetIdx_[code]];
    }

    friend std::ostream& operator<<(std::ostream& os, const Portfolio& port);

    friend class Broker;

  private:
    void registerAssets(Assets assets);
    void registerAssets(Assets assets, std::vector<unsigned int> order);

  private:
    string id_="portfolio_default";
    double initCash_=1'000'000;
    Assets assets_;
    Ledger portfolio_;
    double cash_=initCash_;
    std::unordered_map<string, unsigned int> assetIdx_;
      };


} /*namespace madigan*/

#endif /*  PORTFOLIO_H_ */
