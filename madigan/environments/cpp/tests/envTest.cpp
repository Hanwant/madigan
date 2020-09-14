#include <iostream>

#include "DataSource.h"
#include "Assets.h"
#include "Portfolio.h"
#include "Account.h"
#include "Broker.h"
#include "Env.h"

using namespace madigan;

int main(){

  Assets assets;
  for (auto name: {"sine1", "sine2", "sine3", "sine4"}){
    assets.push_back(Asset(name));
  }
  vector<double> _freq{1., 0.3, 2., 0.5};
  vector<double> _mu{2., 2.1, 2.2, 2.3};
  vector<double> _amp{1., 1.2, 1.3, 1.};
  vector<double> _phase{0., 1., 2., 1.};
  double _dX = 0.01;

  Synth dataSource_ = Synth(_freq, _mu, _amp, _phase, _dX);
  Synth dataSource = Synth();
  Portfolio port = Portfolio("port_0", assets, 1'000'000);

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

  int nAssets = 4;
  std::cout<< "A \n";
  Broker broker1 = Broker(assets, 1'00'000);
  std::cout<< "B \n";
  Portfolio portfolio1 = Portfolio(assets, 1'000'000);
  Portfolio portfolio2 = Portfolio("Portfolio_Test", assets, 1'000'00);
  std::cout<< "C \n";
  Account account1 = Account(assets, 1'000'000);
  std::cout<< "D\n";
  Account account2 = Account("Account_Test", assets, 1'000'000);
  std::cout<< "E \n";
  Account account3 = Account(portfolio1);
  std::cout<< "F \n";

  Env env = Env(&dataSource, broker1);

  return 0;
}

