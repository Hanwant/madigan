#include "Portfolio.h"
// #include <boost/math/special_functions/sign.hpp>
#include <cmath>

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
    assets_ = other.assets_;
    assetsRegistered = other.assetsRegistered;
    cash_ = other.cash_;
    ledger_ = other.ledger_;
    usedMargin_ = other.usedMargin_;
    meanEntryPrices_ = other.meanEntryPrices_;
    borrowedMargin_ = other.borrowedMargin_;
    requiredMargin_ = other.requiredMargin_;
    maintenanceMargin_ = other.maintenanceMargin_;
    borrowedMarginRatio_ = other.borrowedMarginRatio_;

    assetIdx_ = other.assetIdx_;
    registeredDataSource = other.registeredDataSource;
    defaultPrices_ = other.defaultPrices_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           defaultPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
    dataSource_ = other.dataSource_;
    // std::cout << "PORT COPY CONSTR: " << id_ << "\n";
  }
  Portfolio& Portfolio::operator=(const Portfolio& other){
    id_ = other.id_;
    initCash_ = other.initCash_;
    assets_ = other.assets_;
    assetsRegistered = other.assetsRegistered;
    cash_ = other.cash_;
    ledger_ = other.ledger_;
    usedMargin_ = other.usedMargin_;
    meanEntryPrices_ = other.meanEntryPrices_;
    borrowedMargin_ = other.borrowedMargin_;
    requiredMargin_ = other.requiredMargin_;
    maintenanceMargin_ = other.maintenanceMargin_;
    borrowedMarginRatio_ = other.borrowedMarginRatio_;

    assetIdx_ = other.assetIdx_;
    registeredDataSource = other.registeredDataSource;
    defaultPrices_ = other.defaultPrices_;
    if(registeredDataSource){
      new (&currentPrices_) PriceVectorMap(other.currentPrices_.data(),
                                           defaultPrices_.size());
    }
    else{
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
    dataSource_ = other.dataSource_;
    // std::cout << "PORT COPY ASSIGN: " << id_ << "\n";
    return *(this);
  }

  void Portfolio::registerAssets(Assets assets){
    if (assetsRegistered){
      throw std::logic_error("attempting to register assets again");
    }
    for (unsigned int i=0; i<assets.size(); i++){
      string code = assets[i].code;
      if (assetIdx_.find(code) == assetIdx_.end()){
        assetIdx_[code] = i;
        defaultPrices_.push_back(0.);
      }
      else{
        throw std::logic_error(std::string("duplicate asset code ")+code);
      }
    }
    if (!registeredDataSource){
      new (&currentPrices_) PriceVectorMap(defaultPrices_.data(), defaultPrices_.size());
    }
    ledger_ = Ledger::Zero(assets.size());
    meanEntryPrices_ = Ledger::Zero(assets.size());
    borrowedMargin_ = Ledger::Zero(assets.size());
    assetsRegistered = true;
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

  Ledger Portfolio::ledgerNormed() const{
    return (ledger_.array() / equity()).matrix();
  }

  double Portfolio::pnl() const{
    // std::cout <<"eq: " << equity() <<"\n";
    // std::cout <<"assetValue: " << assetValue() <<"\n";
    // std::cout <<"borrowedMargin: " << borrowedMargin_ <<"\n";
    // std::cout <<"borrowedMarginRatio: " << borrowedMarginRatio_ <<"\n";
    // return assetValue() - borrowedMargin_ * borrowedMarginRatio_;
    // double ans = assetValue() - meanEntryPrices_.dot(ledger_);
    return assetValue() - meanEntryPrices_.dot(ledger_);
  }

  double Portfolio::balance() const{
    return cash_ + borrowedAssetValue();
  }

  double Portfolio::usedMargin() const{
    // return requiredMargin_ * meanEntryPrices_.dot(ledger_);
    // return (borrowedMargin_.array() * borrowedMarginRatio()).sum();
    Eigen::ArrayXd longMask = (ledger_.array() > 0.).cast<double>();
    auto longPrices = (meanEntryPrices_.array() * longMask).matrix();
    return requiredMargin_ * ledger_.dot(longPrices);

  }

  double Portfolio::borrowedMargin() const{
    // Eigen::ArrayXd longMask = (ledger_.array() > 0.).cast<double>();
    // auto longPrices = (meanEntryPrices_.array() * longMask).matrix();
    // return (1-requiredMargin_)*ledger_.dot(longPrices);
    return borrowedMargin_.sum();
  }

  const Ledger& Portfolio::borrowedMarginLedger() const{
    return borrowedMargin_;
  }

  double Portfolio::borrowedAssetValue() const{
    // auto shortMask = ledger_.array() <= 0.;
    Eigen::ArrayXd shortMask = (ledger_.array() < 0.).cast<double>();
    auto shortPrices = (currentPrices_.array() * shortMask).matrix();
    return ledger_.dot(shortPrices);
  }

  double Portfolio::availableMargin() const{
    return (balance()+pnl()) / requiredMargin_;
  }

  double Portfolio::borrowedEquity() const{
    return borrowedMargin() - 2*borrowedAssetValue();
  }

  double Portfolio::equity() const{
    return cash_ + assetValue() - borrowedMargin();
  };


  void Portfolio::handleTransaction(string asset, double transactionPrice, double units,
                                    double transactionCost){
    int assetIdx = assetIdx_[asset];
    handleTransaction(assetIdx, transactionPrice, units, transactionCost);
  }

  RiskInfo Portfolio::checkRisk() const{
    // return (equity()<maintenanceMargin_*pnl())? RiskInfo::margin_call: RiskInfo::green
    if (equity() <= -(maintenanceMargin_ * pnl())){
      return RiskInfo::margin_call;
    }
    return RiskInfo::green;
  }
  RiskInfo Portfolio::checkRisk(double amount) const{
    if (equity() <= -(maintenanceMargin_ * pnl())){
      return RiskInfo::margin_call;
    }
    if (availableMargin() <= abs(amount)){
      return RiskInfo::insuff_margin;
    }
    return RiskInfo::green;
  }
  RiskInfo Portfolio::checkRisk(int assetIdx, double units) const{
    double amount = currentPrices_(assetIdx)*units;
    return checkRisk(amount);
  }
  RiskInfo Portfolio::checkRisk(string assetCode, double units) const{
    double amount = currentPrices_(assetIdx_.at(assetCode))*units;
    return checkRisk(amount);
  }

  void Portfolio::handleTransaction(int assetIdx, double transactionPrice, double units,
                                    double transactionCost){

    // Keeping track of average entry price - for pnl/position calculation
    double prevUnits = ledger_(assetIdx);
    if (std::signbit(prevUnits) != std::signbit(units)){
      if (abs(units) > abs(prevUnits)){
        meanEntryPrices_(assetIdx) = transactionPrice;
      }
    }else{
      meanEntryPrices_(assetIdx) += // volume weighted average of position entry prices
        (transactionPrice - meanEntryPrices_(assetIdx)) * (units /(units + prevUnits));
    }

    // Accounting
    double amount_in_base_currency = transactionPrice * units;
    double marginToUse = amount_in_base_currency * requiredMargin_;
    double marginToBorrow = amount_in_base_currency - marginToUse;
    double& borrowedMarginRef = borrowedMargin_(assetIdx);
    borrowedMarginRef += marginToBorrow;
    // borrowedMargin_ += borrowedMargin;
    // usedMargin_ += abs(usedMargin);
    cash_ -= (marginToUse +transactionCost);
    ledger_(assetIdx) += units;

    if (abs(ledger_(assetIdx)) < 0.0000001){
      meanEntryPrices_(assetIdx) = 0.;
      if (borrowedMarginRef > 0.){
        cash_ -= borrowedMarginRef;
        borrowedMarginRef = 0.;
      }
    }

    // if (borrowedMargin_<0. and borrowedAssetValue() == 0.){
    if (borrowedMarginRef < 0. ){
      cash_ -= borrowedMarginRef;
      borrowedMarginRef = 0.;
      // usedMargin_=0;
    }
  }



}
