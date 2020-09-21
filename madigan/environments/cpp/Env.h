#ifndef ENV_H_
#define ENV_H_

#include <vector>
#include <memory>

#include "Assets.h"
#include "DataSource.h"
#include "Broker.h"


namespace madigan{

  // template<typename T>
  class Env{
  public:
    // Env(DataSource* dataSource): dataSource(dataSource){
    //   assets = dataSource->assets;
    // };
    inline Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash);
    // Env(DataSource* dataSource, Broker* broker);
    inline SRDI step(); // No action - I.e Hold
    SRDI step(int action); // Single Asset;
    SRDI step(int action, unsigned int assetIdx); // Multiple Assets
    SRDI step(int action, unsigned int assetIdx, string portforlioID); // Multiple portfolios
    SRDI step(int action, unsigned int assetIdx, string portforlioID, string accountID); // Multiple accounts
    SRDI step(ActionVector actions); // Multiple Assets
    SRDI step(ActionVector actions, string portforlioID); // Multiple portfolios
    SRDI step(ActionVector actions, string portforlioID, string accountID); // Multiple accounts
    SRDI step(Order order);
    ~Env(){};

    const DataSource*  dataSource() const { return dataSource_.get(); }
    const Broker*  broker() const { return broker_.get(); }
    const PriceVector& currentData() const { return dataSource_->currentData();}

  private:
    std::unique_ptr<DataSource> dataSource_;
    std::unique_ptr<Broker> broker_;

  };

  Env::Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash)
    : dataSource_(std::move(dataSource))
  {
    broker_ = std::make_unique<Broker>(assets, initCash);
    for (auto& acc: broker_->accounts_){
      acc.setDataSource(dataSource_.get());
    }
  };

  // Env::Env(DataSource* dataSource, Broker* broker): dataSource_(dataSource), broker_(broker){
  //   for (auto acc: broker_->accountBook()){
  //     acc.second->setDataSource(dataSource_);
  //   }
  // };

  SRDI Env::step(){
    // PriceVector currentprices(*(dataSource_->currentData()));
    PriceVector currentprices = dataSource_->currentData();
    double current_eq = broker_->defaultAccount_->equity();
    // double current_eq = broker.defaultAccount_.defaultPortfolio_.equity();
    PriceVector nextprices = dataSource_->getData();
  }

} // namespace madigan

#endif
