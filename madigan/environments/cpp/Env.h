#ifndef ENV_H_
#define ENV_H_

#include <vector>
#include <memory>
#include <stdexcept>
#include <math.h>

#include "Assets.h"
#include "DataSource.h"
#include "Broker.h"

namespace py=pybind11;

namespace madigan{


  class Env{
  public:
    Config config;
  public:
    // Env(DataSource* dataSource): dataSource(dataSource){
    //   assets = dataSource->assets;
    // };
    // inline Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash);
    Env(string sourceType, Assets assets, double initCash): assets_(assets), initCash_(initCash),
                                                            dataSourceType_(sourceType){
      reset(); }

    Env(string sourceType, Assets assets, double initCash, Config config):
      assets_(assets), initCash_(initCash), dataSourceType_(sourceType), config(config){
      reset(); }
    Env(string sourceType, Assets assets, double initCash, pybind11::dict py_config)
      : Env(sourceType, assets, initCash, makeConfigFromPyDict(py_config)){};

    Env(const Env& other)=delete;
    // Env(const Env& other){
    //   if (other.config.size() > 0){
    //     return Env(other.dataSourceType, other._assets, other._initCash);
    //     }
    //   return Env(other.dataSourceType, other.assets, other.initCash, other.config);
    // }

    virtual inline State reset();
    virtual inline SRDI<double> step(); // No action - I.e Hold
    // SRDISingle step(int action); // Single Asset;
    virtual inline SRDISingle step(int assetIdx, double units); // Multiple Assets
    SRDISingle step(string assetCode, double units){ // Multiple Assets
      return step(broker_->assetIdx_.at(assetCode), units);
    }
    SRDISingle step(int action, unsigned int assetIdx, string portforlioID); // Multiple portfolios
    SRDISingle step(int action, unsigned int assetIdx, string portforlioID, string accountID); // Multiple accounts
    virtual inline SRDIMulti step(const AmountVector& units); // Multiple Assets
    SRDIMulti step(ActionVector actions, string accID); // specific acc, default port
    SRDIMulti step(ActionVector actions, string portforlioID, string accountID); // specific acc, specific port
    SRDISingle step(Order order);
    ~Env(){};

    const DataSource*  dataSource() const { return dataSource_.get(); }
    const Broker*  broker() const { return broker_.get(); }
    const Account*  account() const { return defaultAccount_; }
    const Account*  account(string accID) const { return broker_->accountBook_.at(accID); }
    const Portfolio*  portfolio() const { return defaultPortfolio_; }
    // const Portfolio*  portfolio(string accID) const { return defaultAccount_->portfolioBook_.at(accID); }
    std::size_t currentTime() const {return dataSource_->currentTime(); }
    const PriceVectorMap& currentPrices() const { return currentPrices_;}
    const LedgerMap& ledger() const { return defaultLedger_;}
    Ledger ledgerNormed() const { return defaultPortfolio_->ledgerNormed();}
    const Ledger&  meanEntryPrices() const { return defaultPortfolio_->meanEntryPrices(); }

    string dataSourceType() const { return dataSourceType_; }
    int nAssets() const { return assets_.size(); }
    Assets assets() const { return assets_; }
    double initCash() const { return initCash_; }
    double cash() const { return defaultPortfolio_->cash(); }
    double assetValue() const{ return defaultPortfolio_->assetValue(); }
    AmountVector positionValues() const { return defaultPortfolio_->positionValues(); }
    double equity() const { return defaultPortfolio_->equity(); }
    double usedMargin() const{ return defaultPortfolio_->usedMargin();}
    double availableMargin() const{ return defaultPortfolio_->availableMargin();}
    double requiredMargin() const{ return defaultPortfolio_->requiredMargin();}
    double maintenanceMargin() const{ return defaultPortfolio_->maintenanceMargin();}
    double borrowedMargin() const { return defaultPortfolio_->borrowedMargin(); }
    double borrowedAssetValue() const { return defaultPortfolio_->borrowedAssetValue(); }
    double pnl() const { return defaultPortfolio_->pnl(); }
    AmountVector pnlPositions() const { return defaultPortfolio_->pnlPositions(); }

    void setRequiredMargin(double reqM){
      requiredMargin_ = reqM;
      broker_->setRequiredMargin(reqM);
    }
    void setMaintenanceMargin(double mainM){
      maintenanceMargin_ = mainM;
      broker_->setMaintenanceMargin(mainM);
    }
    void setSlippage(double slippagePct=0., double slippageAbs=0.){
      slippagePct_ = slippagePct;
      slippageAbs_ = slippageAbs;
      broker_->setSlippage(slippagePct_, slippageAbs_);
    }
    void setTransactionCost(double transactionPct=0., double transactionAbs=0.){
      transactionPct_ = transactionPct;
      transactionAbs_ = transactionAbs;
      broker_->setTransactionCost(transactionPct_, transactionAbs_);
    }

    RiskInfo checkRisk(){ return defaultPortfolio_->checkRisk(); }


  private:
    inline void initAccountants();

  private:
    string dataSourceType_;
    Assets assets_;
    double initCash_;
    std::unique_ptr<DataSource> dataSource_;
    std::unique_ptr<Broker> broker_;
    PriceVectorMap currentPrices_{nullptr, 0};
    Portfolio* defaultPortfolio_{nullptr};
    Account* defaultAccount_{nullptr};
    LedgerMap defaultLedger_{nullptr, 0};

    double requiredMargin_{0};
    double maintenanceMargin_{0};
    double slippagePct_{0.};
    double slippageAbs_{0.};
    double transactionPct_{0.};
    double transactionAbs_{0.};
  };

  void Env::initAccountants(){
    // Call after assets_, initCash_ and dataSource_ have been initialized
    broker_ = std::make_unique<Broker>(assets_, initCash_);
    broker_->setDataSource(dataSource_.get());
    broker_->setRequiredMargin(requiredMargin_);
    broker_->setMaintenanceMargin(maintenanceMargin_);
    broker_->setTransactionCost(transactionPct_, transactionAbs_);
    broker_->setSlippage(slippagePct_, slippageAbs_);
    defaultAccount_ = broker_->defaultAccount_;
    defaultPortfolio_ = broker_->defaultPortfolio_;
    const auto& prices = dataSource_->getData();
    new (&currentPrices_) PriceVectorMap(prices.data(), prices.size());
    new (&defaultLedger_) LedgerMap(defaultPortfolio_->ledger().data(),
                                    defaultPortfolio_->ledger().size());
  }

  State Env::reset(){
    if (config.size() > 0){
      dataSource_ = makeDataSource(dataSourceType_, config);
    }
    else dataSource_ = makeDataSource(dataSourceType_);

    initAccountants();
    return State(currentPrices(), ledger(), currentTime());
  }

  SRDISingle Env::step(){
    double prevEq = defaultPortfolio_->equity();
    PriceVector newPrices = dataSource_->getData();
    double currentEq = defaultPortfolio_->equity();
    double reward = log(std::max(currentEq / prevEq, 0.3));
    RiskInfo risk = broker_->checkRisk();
    bool done = (risk == RiskInfo::green)? false: true;
    if (defaultPortfolio_->equity() < 0.1*initCash_){
      done = true;
    }
    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(),
                                 currentTime()}, reward, done, EnvInfo<double>());
  }

  SRDIMulti Env::step(const AmountVector& units){

    double prevEq = broker_->defaultPortfolio_->equity();

    PriceVector newPrices = dataSource_->getData();

    double currentEq = broker_->defaultPortfolio_->equity();
    double reward = log(std::max(currentEq / prevEq, 0.3));

    BrokerResponseMulti response = broker_->handleTransaction(units);
    bool done{false};
    RiskInfo risk = broker_->checkRisk();
    for(const auto& risk: response.riskInfo){
      if (risk != RiskInfo::green && risk != RiskInfo::insuff_margin){
        done = true;
      }
    }
    if (risk != RiskInfo::green || defaultPortfolio_->equity() < 0.1*initCash_){
      done = true;
    }

    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(),
                                 currentTime()}, reward, done, EnvInfoMulti(response));
  }

  SRDISingle Env::step(int assetIdx, double units){

    double prevEq = broker_->defaultPortfolio_->equity();

    PriceVector newPrices = dataSource_->getData();

    double currentEq = broker_->defaultPortfolio_->equity();
    double reward = log(std::max(currentEq / prevEq, 0.3));

    BrokerResponseSingle response = broker_->handleTransaction(assetIdx, units);

    RiskInfo risk = broker_->checkRisk();
    bool done{false};
    if (risk != RiskInfo::green ||
        (response.riskInfo != RiskInfo::green && response.riskInfo != RiskInfo::insuff_margin)){
      done = true;
    }
    if (defaultPortfolio_->equity() < 0.1*initCash_){
      done = true;
    }

    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(),
                                 currentTime()}, reward, done, EnvInfoSingle(response));
  }

} // namespace madigan

#endif
