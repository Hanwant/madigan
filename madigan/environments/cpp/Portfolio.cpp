#include "Portfolio.h"

namespace madigan{

  Portfolio::Portfolio(Assets assets, double initCash): assets_(assets),
                                                        initCash_(initCash),
                                                        cash_(initCash){

    this->portfolio_ = Ledger(assets.size());
    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       Assets assets,
                       double initCash): id_(id), assets_(assets),
                                         initCash_(initCash), cash_(initCash){
    this->portfolio_ = Ledger(assets.size());
    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       Assets assets,
                       double initCash,
                       Ledger portfolio): id_(id), assets_(assets),
                                                       initCash_(initCash), cash_(initCash),
                                                       portfolio_(portfolio){
    registerAssets(assets);
  };
  Portfolio::Portfolio(string id,
                       std::vector<string> assets,
                       double initCash):
    id_(id), assets_(assets), initCash_(initCash), cash_(initCash){
    this->portfolio_ = Ledger(assets.size());
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
    for (unsigned int i=0; i<assets.size(); i++){
      string code = assets[i].code;
      if (assetIdx_.find(code) == assetIdx_.end()){
        assetIdx_[code] = i;
      }
      else{
        throw "asset code already exists";
      }
    }
  }

  double Portfolio::equity() const{
    double sum=0;
    // for(int i=0; i<portfolio_.size(); i++){
    //   sum += (portfolio_[i] * currentPrices_[i]);
    // }
    sum += cash_;
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
}
