#ifndef BROKER_H_
#define BROKER_H_

#include <map>
#include <string>
#include <memory>
#include <iostream>

#include "Portfolio.h"
#include "Account.h"

namespace madigan{
  typedef std::unordered_map<std::string, Account> AccountBook;
  class Broker{
  public:
    Broker(){};
    Broker(Account account);
    Broker(Portfolio portfolio);
    Broker(Assets assets, double initCash);
    Broker(string AccId, Assets assets, double initCash);
    ~Broker(){};

    bool addAccount(Account account);
    void setDefaultAccount(string accId);
    void setDefaultAccount(Account &account);

  private:
    // std::map<std::string, std::shared_ptr<Account>> accountBook;
    AccountBook accountBook;
    Account* defaultAccount;
  };

}

#endif
