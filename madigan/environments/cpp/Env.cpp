#include "Env.h"

namespace madigan{


  // Env::Env(std::unique_ptr<DataSource> dataSource, Assets assets, double initCash)
  //   : dataSource_(std::move(dataSource))
  // {
  //   broker_ = std::make_unique<Broker>(assets, initCash);
  //   for (auto& acc: broker_->accounts_){
  //     acc.setDataSource(dataSource_.get());
  //   }
  // };

  // // Env::Env(DataSource* dataSource, Broker* broker): dataSource_(dataSource), broker_(broker){
  // //   for (auto acc: broker_->accountBook()){
  // //     acc.second->setDataSource(dataSource_);
  // //   }
  // // };

  // SRDI Env::step(){
  //   // PriceVector currentprices(*(dataSource_->currentData()));
  //   PriceVector currentprices = dataSource_->currentData();
  //   double current_eq = broker_->defaultAccount_->equity();
  //   // double current_eq = broker.defaultAccount_.defaultPortfolio_.equity();
  //   PriceVector nextprices = dataSource_->getData();
  // }


}
