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
                       Ledger ledger): id_(id), assets_(assets),
                                                       initCash_(initCash), cash_(initCash),
                                                       ledger_(ledger){
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
      ledger_(portfolio){
    registerAssets(assets);
  };

  Portfolio::Portfolio(const Portfolio& other){
    id_ = other.id_;
    initCash_ = other.initCash_;
    cash_ = other.cash_;
    borrowedMargin_ = other.borrowedMargin_;
    assets_ = other.assets_;
    assetIdx_ = other.assetIdx_;
    ledger_ = other.ledger_;
    registeredDataSource = other.registeredDataSource;
    dataSource_ = other.dataSource_;
    defaultPrices_ = other.defaultPrices_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           defaultPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
    // std::cout << "PORT COPY CONSTR: " << id_ << "\n";
  }
  Portfolio& Portfolio::operator=(const Portfolio& other){
    id_ = other.id_;
    initCash_ = other.initCash_;
    cash_ = other.cash_;
    borrowedMargin_ = other.borrowedMargin_;
    assets_ = other.assets_;
    assetIdx_ = other.assetIdx_;
    ledger_ = other.ledger_;
    registeredDataSource = other.registeredDataSource;
    dataSource_ = other.dataSource_;
    defaultPrices_ = other.defaultPrices_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           defaultPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
    // std::cout << "PORT COPY ASSIGN: " << id_ << "\n";
    return *(this);
  }

  void Portfolio::registerAssets(Assets assets){
    this->ledger_ = Ledger::Zero(assets.size());
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

  std::ostream& operator<<(std::ostream& os, const Portfolio& port){
    os << "Port id: " << port.id() << " equity: " << port.equity() << " cash: " << port.cash() << "\n";
    return os;
  };

  std::unordered_map<string, unsigned int> Portfolio::assetIdx() const{
    return assetIdx_;
  }
  unsigned int Portfolio::assetIdx(const string code) const {
    return ledger_(assetIdx_.find(code)->second);
  }

  double Portfolio::equity() const{
    double sum=0;
    sum += cash_;
    sum += assetValue();
    sum -= borrowedMargin_;
    return sum;
  };

  Ledger Portfolio::ledgerNormed() const{
    return (ledger_.array() / equity()).matrix();
  }

  double Portfolio::availableMargin() const{
    return cash_ - borrowedMargin_;
  }

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
    ledger_[assetIdx] += units;
    if (borrowedMargin_<0.){
      cash_ -= borrowedMargin_;
      borrowedMargin_=0.;
    }
  }


}
