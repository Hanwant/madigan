#include <iostream>

#include "Env.h"
#include "DataSource.h"
#include "Portfolio.h"
#include "Broker.h"

using namespace madigan;

int main(){

  Env env = Env();
  Broker broker = Broker();
  vector<double> _freq{1., 0.3, 2., 0.5};
  vector<double> _mu{2., 2.1, 2.2, 2.3};
  vector<double> _amp{1., 1.2, 1.3, 1.};
  vector<double> _phase{0., 1., 2., 1.};
  double _dX = 0.01;

  Synth dataSource_ = Synth(_freq, _mu, _amp, _phase, _dX);
  Synth dataSource = Synth();
  Portfolio port = Portfolio("port_0", 4, 1'000'000);

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


  std::cout << "Done Init \n";

  return 0;
}
