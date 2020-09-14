#include "Broker.h"


namespace madigan {
  Broker::Broker(Assets assets, double initCash){
    Account account(assets, initCash);
    addAccount(account);
  }
  Broker::Broker(string id, Assets assets, double initCash){
    Account account(id, assets, initCash);
    addAccount(account);
  }
  Broker::Broker(Account account){
    addAccount(account);
  }
  Broker::Broker(Portfolio portfolio){
    Account account(portfolio);
    addAccount(account);
  }

  bool Broker::addAccount(Account &account){
    if (accountBook.find(account.id()) == accountBook.end()){
      // accountBook[account.id()] = std::shared_ptr<Account>(&account);
      accountBook[account.id()] = account;
      return true;
    }
    else{
      return false;
    }
  }
}
