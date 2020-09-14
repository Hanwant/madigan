#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>
#include <memory>
#include <iostream>

#include "Portfolio.h"
#include "Account.h"

namespace madigan{
  class Broker{
  public:
    Broker(Assets assets, double initCash);
    Broker(string id, Assets assets, double initCash);
    Broker(Account account);
    Broker(Portfolio portfolio);
    ~Broker(){};

    bool addAccount(Account &account);
  private:
    // std::map<std::string, std::shared_ptr<Account>> accountBook;
    std::unordered_map<std::string, Account> accountBook;
  };

}

#endif
