#ifndef ENV_H_
#define ENV_H_

#include <vector>

#include "Assets.h"
#include "DataSource.h"
#include "Broker.h"


namespace madigan{

  #ifndef Order
    struct Order{};
  #endif



  class Env{
  public:
    // Env(DataSource* dataSource): dataSource(dataSource){
    //   assets = dataSource->assets;
    // };
    Env(DataSource* dataSource, Broker* broker);
    SRDI step(); // No action - I.e Hold
    SRDI step(int action); // Single Asset;
    SRDI step(int action, unsigned int assetIdx); // Multiple Assets
    SRDI step(int action, unsigned int assetIdx, string portforlioID); // Multiple portfolios
    SRDI step(int action, unsigned int assetIdx, string portforlioID, string accountID); // Multiple accounts
    SRDI step(ActionVector actions); // Multiple Assets
    SRDI step(ActionVector actions, string portforlioID); // Multiple portfolios
    SRDI step(ActionVector actions, string portforlioID, string accountID); // Multiple accounts
    SRDI step(Order order);
    ~Env(){};

    PriceVector currentPrices(){ return dataSource_->currentData();}

  private:
    DataSource* dataSource_;
    Broker* broker_;

  };
} // namespace madigan

#endif
