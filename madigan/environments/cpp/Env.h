#ifndef ENV_H_
#define ENV_H_

#include <vector>

#include "Assets.h"
#include "Broker.h"
#include "DataSource.h"


namespace madigan{
  class Env{
  public:
    Env(DataSource* dataSource): dataSource(dataSource){
      assets = dataSource->assets;
    };
    Env(DataSource* dataSource, Broker broker): dataSource(dataSource), broker(broker){};
    ~Env(){};

  private:
    Assets assets{};
    DataSource* dataSource;
    Broker broker{assets, 1'000'000};

  };
} // namespace madigan

#endif 
