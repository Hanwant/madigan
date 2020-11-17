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
    return (positionValues().array() / equity()).matrix();
  }

  Ledger Portfolio::ledgerAbsNormed() const{
    Ledger ledgerNorm = ledgerNormed();
    return (ledgerNorm.array() /
            ledgerNorm.array().abs().sum()).matrix();
  }

  Ledger Portfolio::ledgerNormedFull() const{
    Ledger led = Ledger(nAssets()+1);
    led << (cash_-borrowedMargin())/equity(),
      (positionValues().array() / equity()).matrix();
    return led;
  }

  Ledger Portfolio::ledgerAbsNormedFull() const{
    Ledger ledgerNormFull = ledgerNormedFull();
    return (ledgerNormFull.array() /
            ledgerNormFull.array().abs().sum()).matrix();
  }

  double Portfolio::pnl() const{
    return assetValue() - meanEntryPrices_.dot(ledger_);
  }

  AmountVector Portfolio::pnlPositions() const{
    return positionValues().array() - meanEntryPrices_.cwiseProduct(ledger_).array();
  }

  double Portfolio::balance() const{
    Eigen::ArrayXd shortMask = (ledger_.array() < 0.).cast<double>();
    auto shortPrices = (meanEntryPrices_.array() * shortMask).matrix();
    double borrowedAssetEntryValue = ledger_.dot(shortPrices);
    return cash_ + borrowedAssetEntryValue;
  }

  double Portfolio::usedMargin() const{
    return requiredMargin_ * ledger_.cwiseAbs().dot(meanEntryPrices_);

  }

  double Portfolio::borrowedMargin() const{
    return borrowedMargin_.sum();
  }

  double Portfolio::equity() const{
    return cash_ + assetValue() - borrowedMargin();
  }

  const Ledger& Portfolio::borrowedMarginLedger() const{
    return borrowedMargin_;
  }

  double Portfolio::borrowedAssetValue() const{
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

  void Portfolio::handleTransaction(string asset, double transactionPrice, double units,
                                    double transactionCost){
    int assetIdx = assetIdx_[asset];
    handleTransaction(assetIdx, transactionPrice, units, transactionCost);
  }

  RiskInfo Portfolio::checkRisk() const{
    double marginRequired = maintenanceMargin_ * pnl();
    if (equity() <= -marginRequired){
      return RiskInfo::margin_call;
    }
    if ((balance()+pnl()) <= -marginRequired){
      return RiskInfo::margin_call;
    }
    return RiskInfo::green;
  }

  RiskInfo Portfolio::checkRisk(int assetIdx, double units) const{
    double cashAmount = currentPrices_(assetIdx)*units;
    const double& currentUnits = ledger_(assetIdx);
    if (std::signbit(units) != std::signbit(currentUnits)){
      if (units > -1*currentUnits){ // attempting to reverse position
        double excess = units + currentUnits;
        if (availableMargin() <= abs(currentPrices_[assetIdx]*excess) ||
            balance() <= 0.){
          return RiskInfo::insuff_margin; // CANT REVERSE OR CLOSE POSITION - NO PARTIAL CLOSE
        }
      }
      return RiskInfo::green;
    }
    else{
      if (checkRisk() == RiskInfo::margin_call){
        return RiskInfo::margin_call;
      }
      else{
        if (availableMargin() <= abs(cashAmount)
            || balance() <= 0.){
          return RiskInfo::insuff_margin;
        }
        return RiskInfo::green;
      }
    }
  }
  RiskInfo Portfolio::checkRisk(string assetCode, double units) const{
    return checkRisk(assetIdx_.at(assetCode), units);
  }

  void Portfolio::handleTransaction(int assetIdx, double transactionPrice, double units,
                         double transactionCost){
    // Keeping track of average entry price - for pnl/position calculation
    double& currentUnits = ledger_(assetIdx);
    double& meanEntryPrice = meanEntryPrices_(assetIdx);

    // std::cout << "==========================================================\n";
    // std::cout << "----------------------------------------------------------\n";
    // std::cout << "Pre Transaction\n";
    // std::cout << "Current Units " << currentUnits << "\n";
    // std::cout << "units to buy " << units << "\n";
    // std::cout << "transactionPrice " << transactionPrice<< "\n";
    // std::cout << "meanEntryPrice " << meanEntryPrice << "\n";
    // std::cout << "sign bits: "<< std::signbit(currentUnits) << ", "<< std::signbit(units) << "\n";

    if (std::signbit(currentUnits) != std::signbit(units)){
      if (abs(units) > abs(currentUnits)){
        // CLOSE POSITION
        units += currentUnits; // take away amount required to close position
        cash_ += currentUnits*transactionPrice; // do cash accounting first to close position
        currentUnits = 0.; // explicitly close position by setting ledger_(assetIdx) to 0.
        // SET NEW MEAN ENTRY PRICE
        meanEntryPrice = transactionPrice;
        // std::cout<< "reversed and set mean entry to new: " << meanEntryPrice << "\n";
      }
    }
    else{
      // std::cout<< "accumulating trans, mean, unit, curr: ";
      // std::cout<< transactionPrice << ", ";
      // std::cout<< meanEntryPrice<< ", ";
      // std::cout<< units << ", ";
      // std::cout<< currentUnits<< ", \n";
      meanEntryPrice += // volume weighted average of position entry prices
        (transactionPrice - meanEntryPrice) * (units /(units + currentUnits));
      // std::cout << "accum numer, denom : "<<(transactionPrice - meanEntryPrice) <<", "<< (units /(units + currentUnits));
      // std::cout <<"\n";
      // std::cout<< "accumulated and set mean entry to new: " << meanEntryPrice << "\n";
    }
    // Accounting
    double amount_in_base_currency = transactionPrice * units;
    double marginToUse = amount_in_base_currency * requiredMargin_;
    double marginToBorrow = amount_in_base_currency - marginToUse;
    double& borrowedMarginRef = borrowedMargin_(assetIdx);

    borrowedMarginRef += marginToBorrow;
    cash_ -= (marginToUse +transactionCost);
    currentUnits += units;

    // std::cout << "----------------------------------------------------------\n";
    // std::cout << "Post Transaction\n";
    // std::cout << "Current Units " << currentUnits << "\n";
    // std::cout << "units to buy " << units << "\n";
    // std::cout << "meanEntryPrice " << meanEntryPrice << "\n";

    if (abs(currentUnits) < 0.0000001){
      meanEntryPrice = 0.;
      if (borrowedMarginRef > 0.){
        cash_ -= borrowedMarginRef;
        borrowedMarginRef = 0.;
      }
    }
    if (borrowedMarginRef < 0. ){
      cash_ -= borrowedMarginRef;
      borrowedMarginRef = 0.;
    }
  }

  // void Portfolio::handleTransaction(int assetIdx, double transactionPrice, double units,
  //                        double transactionCost){
  //   // Keeping track of average entry price - for pnl/position calculation
  //   double& currentUnits = ledger_(assetIdx);
  //   double& meanEntryPrice = meanEntryPrices_(assetIdx);

  //   // std::cout << "Current Units " << currentUnits << "\n";
  //   // std::cout << "units to buy " << units << "\n";
  //   // std::cout << "meanEntryPrice" << meanEntryPrice << "\n";

  //   if (std::signbit(currentUnits) != std::signbit(units)){
  //     if (abs(units) > abs(currentUnits)){
  //       units += currentUnits;
  //       close(assetIdx, transactionPrice, 0.);
  //     }
  //   }
  //   else{
  //     meanEntryPrice += // volume weighted average of position entry prices
  //       (transactionPrice - meanEntryPrice) * (units /(units + currentUnits));
  //   }
  //   // Accounting
  //   double amount_in_base_currency = transactionPrice * units;
  //   double marginToUse = amount_in_base_currency * requiredMargin_;
  //   double marginToBorrow = amount_in_base_currency - marginToUse;
  //   double& borrowedMarginRef = borrowedMargin_(assetIdx);
  //   borrowedMarginRef += marginToBorrow;
  //   cash_ -= (marginToUse +transactionCost);
  //   currentUnits += units;

  //   if (abs(currentUnits) < 0.0000001){
  //     meanEntryPrice = 0.;
  //     if (borrowedMarginRef > 0.){
  //       cash_ -= borrowedMarginRef;
  //       borrowedMarginRef = 0.;
  //     }
  //   }
  //   if (borrowedMarginRef < 0. ){
  //     cash_ -= borrowedMarginRef;
  //     borrowedMarginRef = 0.;
  //   }

  // }
  void Portfolio::close(int assetIdx, double transactionPrice,
                        double transactionCost){
    double& currentUnits = ledger_(assetIdx);
    // std::cout << "CLOSING\n";
    // std::cout << "transPrice: " << transactionPrice << "\n";
    if (currentUnits != 0.){
      handleTransaction(assetIdx, transactionPrice, -1*currentUnits,
                        transactionCost);
    }
  }

  // void Portfolio::handleTransaction(int assetIdx, double transactionPrice, double units,
  //                                   double transactionCost){

  //   // Keeping track of average entry price - for pnl/position calculation
  //   double& prevUnits = ledger_(assetIdx);
  //   if(units>0.){
  //     if (prevUnits>0.){
  //       meanEntryPrices_(assetIdx) += // volume weighted average of position entry prices
  //         (transactionPrice - meanEntryPrices_(assetIdx)) * (units /(units + prevUnits));
  //       double amount_in_base_currentcy = transactionPrice*units;
  //       double marginToUse = amount_in_base_currency*requiredMargin_;
  //       double marginToBorrow = amount_in_base_currency - marginToUse;
  //       double& borrowedMarginRef = borrowedMargin_(assetIdx);
  //       borrowedMarginRef += marginToBorrow;
  //       cash_ -= (marginToUse + transacionCost);
  //       prevUnits
  //     }
  //   }

  // }
}
