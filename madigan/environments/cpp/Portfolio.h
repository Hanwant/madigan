#ifndef LEDGER_H_
#define LEDGER_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include<Eigen/Core>

#include "Assets.h"
#include "DataTypes.h"
#include "DataSource.h"


namespace madigan{

  class Account; // forward declare for friend class declaration below

  using std::string;

  class Portfolio {
  public:
    /* Portfolio(){}; */
    Portfolio(Assets assets, double initCash);
    Portfolio(string id, Assets assets, double initCash);
    Portfolio(string id, Assets assets, double initCash, Ledger portfolio);
    Portfolio(string id, std::vector<string> assets, double initCash);
    Portfolio(string id, std::vector<string> assets, double initCash, Ledger portfolio);
    Portfolio(const Portfolio& other);
    Portfolio& operator=(const Portfolio& other);
    Portfolio(const Portfolio&& other)=delete;
    Portfolio& operator=(const Portfolio&& other)=delete;
    ~Portfolio()=default;

    void setDataSource(DataSource* source);

    string id() const { return id_;}
    Assets assets() const {return assets_;}
    std::unordered_map<string, unsigned int> assetIdx() const;
    unsigned int assetIdx(const string code) const;
    const DataSource* dataSource() const{ return dataSource_;}
    const PriceVectorMap& currentPrices() const { return currentPrices_;}

    int nAssets() const { return assets_.size();}
    double initCash() const {return initCash_;}
    double cash() const { return cash_;}
    const Ledger& ledger() const {return ledger_;}
    Ledger ledgerNormed() const;
    /* double usedMargin() const { return usedmargin.sum()} */
    double assetValue() const { return ledger_.dot(currentPrices_); }
    double equity() const;
    double availableMargin() const;
    const double borrowedMargin() const {return borrowedMargin_;}
    double borrowedCash() const;
    double operator[](string code) {
      return ledger_[assetIdx_[code]];
    }
    double operator[](int assetIdx) {
      return ledger_[assetIdx];
    }

    void handleTransaction(string asset, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);
    void handleTransaction(int assetIdx, double tranactionPrice,
                           double units, double transactionCost, double requiredMargin);

    friend std::ostream& operator<<(std::ostream& os, const Portfolio& port);

    friend class Broker;
    friend class Account;

  private:
    void registerAssets(Assets assets);
    void registerAssets(Assets assets, std::vector<unsigned int> order);

  private:
    string id_="ledger_default";
    double initCash_=1'000'000;
    Assets assets_;
    double cash_=initCash_;
    Ledger ledger_;
    Ledger usedMargin_;
    double borrowedMargin_{0.};

    std::unordered_map<string, unsigned int> assetIdx_;
    bool registeredDataSource{false};
    std::vector<double> defaultPrices_; // just to have some default prices (init to 0.) when datasource is na
    PriceVectorMap currentPrices_{nullptr, 0};
    DataSource* dataSource_{nullptr};

      };


} /*namespace madigan*/

#endif /*  LEDGER_H_ */
