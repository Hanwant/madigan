#ifndef ACCOUNT_H_
#define ACCOUNT_H_
#include "Portfolio.h"

namespace madigan{
  class Account{
  public:
    Account(){};
    ~Account(){};
    friend class Broker;
  private:
    Portfolio portfolio;
  };

} // namespace madigan

#endif // ACCOUNT_H_
