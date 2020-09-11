#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>

#include "Portfolio.h"
#include "Account.h"

namespace madigan{
  class Broker{
  public:
    Broker(){};
    ~Broker(){};
  private:
    std::map<std::string, Account> accountBook;
  };

}

#endif
