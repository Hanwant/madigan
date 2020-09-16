#include "Broker.h"


namespace madigan {

  Broker::Broker(Account account){
    addAccount(account);
  }
  Broker::Broker(Portfolio portfolio){
    Account account(portfolio);
    addAccount(account);
  }

  Broker::Broker(Assets assets, double initCash){
    Account account(assets, initCash);
    addAccount(account);
  }
  Broker::Broker(string AccId, Assets assets, double initCash){
    Account account(AccId, assets, initCash);
    addAccount(account);
  }

  bool Broker::addAccount(Account account){
    if (accountBook.find(account.id()) == accountBook.end()){
      accountBook[account.id()] = account;
      if(accountBook.size() == 1){
        setDefaultAccount(account);
      }
      return true;
    }
    else{
      return false;
    }
  }

  void Broker::setDefaultAccount(string accId){
    auto acc = accountBook.find(accId);
    if (acc != accountBook.end()){
      defaultAccount = &(acc->second);
    }
  }
  void Broker::setDefaultAccount(Account &account){
    auto found = accountBook.find(account.id());
    if (found != accountBook.end()){
      defaultAccount = &(found->second);
    }
    else{
      addAccount(account);
      setDefaultAccount(account);
    }

  }
}
