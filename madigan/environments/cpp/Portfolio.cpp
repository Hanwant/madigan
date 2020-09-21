#include "Portfolio.h"

namespace madigan{

  Portfolio::Portfolio(Assets assets, double initCash): assets_(assets),
                                                        initCash_(initCash),
                                                        cash_(initCash){

    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       Assets assets,
                       double initCash): id_(id), assets_(assets),
                                         initCash_(initCash), cash_(initCash){
    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       Assets assets,
                       double initCash,
                       Ledger portfolio): id_(id), assets_(assets),
                                                       initCash_(initCash), cash_(initCash),
                                                       portfolio_(portfolio){
  };
  Portfolio::Portfolio(string id,
                       std::vector<string> assets,
                       double initCash):
    id_(id), assets_(assets), initCash_(initCash), cash_(initCash){
    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       std::vector<string> assets,
                       double initCash,
                       Ledger portfolio)
    : id_(id), assets_(assets), initCash_(initCash), cash_(initCash),
      portfolio_(portfolio){
    registerAssets(assets);
  };

  void Portfolio::registerAssets(Assets assets){
    this->portfolio_ = Ledger::Zero(assets.size());
    usedMargin_ = Ledger::Zero(assets.size());
    for (unsigned int i=0; i<assets.size(); i++){
      string code = assets[i].code;
      if (assetIdx_.find(code) == assetIdx_.end()){
        assetIdx_[code] = i;
        defaultPrices_.push_back(0.);
      }
      else{
        throw "asset code already exists";
      }
    }
    if (!registeredDataSource){
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
  }

  void Portfolio::setDataSource(DataSource* source){
    dataSource_ = source;
    new (&currentPrices_) PriceVectorMap(source->currentData().data(), source->currentData().size());
    registeredDataSource = true;
  }

  double Portfolio::equity() const{
    double sum=0;
    // for(int i=0; i<portfolio_.size(); i++){
    //   sum += (portfolio_(i) * currentPrices_(i));
    // }
    sum += portfolio_.dot(currentPrices_);
    sum += cash_;
    sum -= borrowedMargin_;
    return sum;
  };

  std::unordered_map<string, unsigned int> Portfolio::assetIdx() const{
    return assetIdx_;
  }
  unsigned int Portfolio::assetIdx(const string code) const {
    return portfolio_(assetIdx_.find(code)->second);
  }

  std::ostream& operator<<(std::ostream& os, const Portfolio& port){
    os << "Port id: " << port.id() << " equity: " << port.equity() << "\n";
    return os;
  };


  void Portfolio::handleTransaction(string asset, double transactionPrice, double units,
                                    double transactionCost, double requiredMargin){
    int assetIdx = assetIdx_[asset];
    handleTransaction(assetIdx, transactionPrice, units, transactionCost, requiredMargin);
  }
  void Portfolio::handleTransaction(int assetIdx, double transactionPrice, double units,
                                    double transactionCost, double requiredMargin){
    double amount_in_base_currency = transactionPrice * units;
    double usedMargin = amount_in_base_currency * requiredMargin;
    double borrowedMargin = amount_in_base_currency - usedMargin;
    borrowedMargin_ += borrowedMargin;
    cash_ -= (usedMargin+transactionCost);
    portfolio_[assetIdx] += units;

    if (borrowedMargin_<0.){
      cash_ -= borrowedMargin_;
      borrowedMargin_=0.;
    }
  }

}
