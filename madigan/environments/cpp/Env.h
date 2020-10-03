#ifndef ENV_H_
#define ENV_H_

#include <vector>
#include <memory>
#include <stdexcept>

#include "Assets.h"
#include "DataSource.h"
#include "Broker.h"

namespace py=pybind11;

namespace madigan{


  class Env{
  public:
    Config config;
    string dataSourceType;
  public:
    // Env(DataSource* dataSource): dataSource(dataSource){
    //   assets = dataSource->assets;
    // };
    // inline Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash);
    Env(string sourceType, Assets assets, double initCash): dataSourceType(sourceType){
      reset(); initAccountants(assets, initCash);}

    Env(string sourceType, Assets assets, double initCash, Config config):
      dataSourceType(sourceType), config(config){
      reset(); initAccountants(assets, initCash);}
    Env(string sourceType, Assets assets, double initCash, pybind11::dict py_config)
      : Env(sourceType, assets, initCash, makeConfigFromPyDict(py_config)){};

    Env(const Env& other)=delete;
    virtual inline void reset();
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
    const Account*  account(string accID) const { return broker_->acountBook_.at(accID); }
    const Portfolio*  portfolio() const { return defaultPortfolio_; }
    const Portfolio*  portfolio(string accID) const { return defaultAccount_->portfolioBook_.at(accID); }
    const PriceVectorMap& currentPrices() const { return currentPrices_;}
    const LedgerMap& ledger() const { return defaultLedger_;}
    Ledger ledgerNormed() const { return defaultPortfolio_->ledgerNormed();}

    int nAssets() const { return broker_->nAssets(); }
    Assets assets() const { return broker_->assets(); }
    double cash() const { return defaultPortfolio_->cash(); }
    double assetValue() const{ return defaultPortfolio_->assetValue(); }
    double equity() const { return defaultPortfolio_->equity(); }
    double usedMargin() const{ return defaultPortfolio_->usedMargin();}
    double availableMargin() const{ return defaultPortfolio_->availableMargin();}
    double requiredMargin() const{ return defaultPortfolio_->requiredMargin();}
    double maintenanceMargin() const{ return defaultPortfolio_->maintenanceMargin();}
    double borrowedMargin() const { return defaultPortfolio_->borrowedMargin(); }
    double borrowedAssetValue() const { return defaultPortfolio_->borrowedAssetValue(); }
    double pnl() const { return defaultPortfolio_->pnl(); }

    void setRequiredMargin(double reqM){ broker_->setRequiredMargin(reqM); }
    void setMaintenanceMargin(double mainM){ broker_->setMaintenanceMargin(mainM); }


  private:
    inline void initAccountants(Assets assets, double initCash);

  private:
    std::unique_ptr<DataSource> dataSource_;
    std::unique_ptr<Broker> broker_;
    PriceVectorMap currentPrices_{nullptr, 0};
    Portfolio* defaultPortfolio_{nullptr};
    Account* defaultAccount_{nullptr};
    LedgerMap defaultLedger_{nullptr, 0};
  };

  void Env::initAccountants(Assets assets, double initCash){
    // Call after dataSource_ has been initialized
    broker_ = std::make_unique<Broker>(assets, initCash);
    broker_->setDataSource(dataSource_.get());
    defaultAccount_ = broker_->defaultAccount_;
    defaultPortfolio_ = broker_->defaultPortfolio_;

    const auto& prices = dataSource_->getData();
    new (&currentPrices_) PriceVectorMap(prices.data(), prices.size());
    new (&defaultLedger_) LedgerMap(defaultPortfolio_->ledger().data(),
                                    defaultPortfolio_->ledger().size());
  }

  void Env::reset(){
    if(dataSourceType == "Synth"){
      if (config.size() > 0){
        dataSource_ = make_unique<Synth>(config);
      }
      else{
        dataSource_ = make_unique<Synth>();
      }
    }
    else{
      throw NotImplemented("Only Synth as datasource is implemented");
    }
  }

  SRDISingle Env::step(){
    double prevEq = broker_->defaultAccount_->equity();
    PriceVector newPrices = dataSource_->getData();
    double currentEq = broker_->defaultAccount_->equity();
    double reward = (currentEq-prevEq) / prevEq;
    RiskInfo risk = broker_->checkRisk();
    bool done = (risk == RiskInfo::green)? true: false;
    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(), 0},
                   reward, done, EnvInfo<double>());
  }

  SRDIMulti Env::step(const AmountVector& units){

    double prevEq = broker_->defaultPortfolio_->equity();

    BrokerResponseMulti response = broker_->handleTransaction(units);
    PriceVector newPrices = dataSource_->getData();

    double currentEq = broker_->defaultPortfolio_->equity();
    double reward = (currentEq-prevEq) / prevEq;

    bool done{false};
    for(const auto& risk: response.riskInfo){
      if (risk != RiskInfo::green){
        done = true;
      }
    }

    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(), 0},
                           reward, done, EnvInfoMulti(response));
  }

  SRDISingle Env::step(int assetIdx, double units){

    double prevEq = broker_->defaultPortfolio_->equity();
    BrokerResponseSingle response = broker_->handleTransaction(assetIdx, units);
    PriceVector newPrices = dataSource_->getData();

    double currentEq = broker_->defaultPortfolio_->equity();
    double reward = (currentEq-prevEq) / prevEq;
    bool done = (response.riskInfo == RiskInfo::green)? false: true;

    return std::make_tuple(State{newPrices, broker_->defaultAccount_->ledgerNormed(), 0},
                           reward, done, EnvInfoSingle(response));
  }

} // namespace madigan

#endif
