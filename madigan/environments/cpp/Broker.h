#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>

#include "Portfolio.h"

namespace madigan{
  class Broker{
  public:
    Broker();
    ~Broker();
  private:
    std::map<std::string, Portfolio> ports;
    // std::map<string, Account> accounts;
  };

}

#endif
