#ifndef ENV_H_
#define ENV_H_

#include "Broker.h"
#include "DataSource.h"

namespace madigan{
  class Env{
  public:
    Env();
    ~Env();

  private:
    int nAssets;
    DataSource dataSource;
    Broker broker;

  };
} // namespace madigan

#endif 
