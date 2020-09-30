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
    // Env(DataSource* dataSource): dataSource(dataSource){
    //   assets = dataSource->assets;
    // };
    // inline Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash);
    inline Env(string dataSourceType, Assets assets, double initCash);
    inline Env(string dataSourceType, Assets assets, double initCash, Config config);
    inline Env(string dataSourceType, Assets assets, double initCash, pybind11::dict config);
    Env(const Env& other)=delete;
    // Env(DataSource* dataSource, Broker* broker);
    inline SRDI<double> step(); // No action - I.e Hold
    SRDI<double> step(int action); // Single Asset;
    SRDI<double> step(int action, unsigned int assetIdx); // Multiple Assets
    SRDI<double> step(int action, unsigned int assetIdx, string portforlioID); // Multiple portfolios
    SRDI<double> step(int action, unsigned int assetIdx, string portforlioID, string accountID); // Multiple accounts
    SRDI<PriceVector> step(ActionVector actions); // Multiple Assets
    SRDI<PriceVector> step(ActionVector actions, string accID); // specific acc, default port
    SRDI<PriceVector> step(ActionVector actions, string portforlioID, string accountID); // specific acc, specific port
    SRDI<double> step(Order order);
    ~Env(){};

    const DataSource*  dataSource() const { return dataSource_.get(); }
    const Broker*  broker() const { return broker_.get(); }
    const PriceVector& currentData() const { return dataSource_->currentData();}

  private:
    std::unique_ptr<DataSource> dataSource_;
    std::unique_ptr<Broker> broker_;
  };

  Env::Env(string sourceType, Assets assets, double initCash)
  {
    if(sourceType == "Synth"){
      dataSource_ = std::make_unique<Synth>();
    }
    else throw NotImplemented("Only Synth as datasource is implemented");

    broker_ = std::make_unique<Broker>(assets, initCash);
    for (auto& acc: broker_->accounts_){
      acc.setDataSource(dataSource_.get());
    }
  };
  Env::Env(string sourceType, Assets assets, double initCash, Config config)
  {
    if(sourceType == "Synth"){
        dataSource_ = std::make_unique<Synth>(config);
      }
    else throw NotImplemented("only synth as dataource is implemented");

    broker_ = std::make_unique<Broker>(assets, initCash);
    for (auto& acc: broker_->accounts_){
      acc.setDataSource(dataSource_.get());
    }
  };
  Env::Env(string sourceType, Assets assets, double initCash, pybind11::dict py_config)
    : Env(sourceType, assets, initCash, makeConfigFromPyDict(py_config)){};


  SRDI<double> Env::step(){
    double prevEq = broker_->defaultAccount_->equity();
    PriceVector newPrices = dataSource_->getData();
    double currentEq = broker_->defaultAccount_->equity();
    double reward = (currentEq-prevEq) / prevEq;
    RiskInfo risk = broker_->checkRisk();
    bool done = (risk == RiskInfo::green)? true: false;
    return SRDI<double>(State{newPrices, broker_->defaultAccount_->ledgerNormed(), 0},
                   reward, done, EnvInfo<double>());
  }

} // namespace madigan

#endif
