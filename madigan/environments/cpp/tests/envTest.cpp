#include <iostream>

#include "DataSource.h"
#include "Assets.h"
#include "Portfolio.h"
#include "Account.h"
#include "Broker.h"
#include "Env.h"

using namespace madigan;

void testDataSource(){

  vector<double> _freq{1., 0.3, 2., 0.5};
  vector<double> _mu{2., 2.1, 2.2, 2.3};
  vector<double> _amp{1., 1.2, 1.3, 1.};
  vector<double> _phase{0., 1., 2., 1.};
  double _dX = 0.01;

  Synth dataSource_ = Synth(_freq, _mu, _amp, _phase, _dX);
  Synth dataSource = Synth();
  std::cout<< "dataSource 1. \n";
  for (auto dat: dataSource_.getData()){
    std::cout << dat << " ";
  }
  std::cout << "\n";

  std::cout<< "dataSource 2. \n";
  for (auto dat: dataSource.getData()){
    std::cout << dat << " ";
  }
  std::cout << "\n";
}

void testPortfolio(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // for (auto name: {"sine1", "sine2", "sine3", "sine4"}){
  //   assets.push_back(Asset(name));
  // }

  Portfolio portfolio1 = Portfolio();
  Portfolio portfolio2 = Portfolio(assets, 1'000'000);
  Portfolio portfolio3 = Portfolio("Portfolio_Test", assets, 1'000'00);
  std::cout << portfolio1;
  std::cout << portfolio2;
  std::cout << portfolio3;
  std::cout << "showing ledger" << "\n";
  std::cout << portfolio1.portfolio().size() << "\n";
  std::cout << portfolio1.portfolio() << "\n";
}

void testAccount(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Portfolio portfolio1 = Portfolio();
  Account account1 = Account(assets, 1'000'000);
  Account account2 = Account("Account_Test", assets, 1'000'000);
  Account account3 = Account(portfolio1);
  PortfolioBook ports1 = account1.portfolios();
  PortfolioBook ports2 = account2.portfolios();
  PortfolioBook ports3 = account3.portfolios();

  std::vector<PortfolioBook> ports{ports1, ports2, ports3};
  for (auto port: ports){
    for(auto it = port.begin(); it!= port.end(); it++){
      std::cout<<it->second<<"\n";
    }
  }
}

void testBroker(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Account account1(assets, 1'000'000);
  Broker broker1();
  Broker broker2("broker_constructed_acc", assets, 1'000'000);
  Broker broker3(account1);
  Broker broker4(account1.portfolio());

}
void testEnv(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Synth dataSource1 = Synth();
  Broker broker1 = Broker(assets, 1'00'000);
  Env env = Env(&dataSource1, &broker1);
}

int main(){


  testDataSource();
  testPortfolio();
  testAccount();
  testBroker();
  testEnv();



  return 0;
}

