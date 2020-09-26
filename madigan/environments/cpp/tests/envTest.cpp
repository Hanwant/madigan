#include <iostream>
#include <assert.h>
#include <stdexcept>

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
  assert(dataSource.getData().isApprox(dataSource.getData()));
  std::cout << "\n";
}

void testPortfolio(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // for (auto name: {"sine1", "sine2", "sine3", "sine4"}){
  //   assets.push_back(Asset(name));
  // }

  Portfolio portfolio1 = Portfolio(assets, 1'000'000);
  Portfolio portfolio2 = Portfolio("Portfolio_Test", assets, 1'000'000);
  Portfolio portfolio3 = Portfolio("Portfolio_Test",
                                   std::vector<string>{"sine1", "sine2", "sine3", "sine4"}, 1'000'000);
  Synth dataSource = Synth();
  dataSource.getData();
  for (auto port: {portfolio1, portfolio2, portfolio3}){
    std::cout<< "===========================================\n";
    std::cout << "testing port" << std::endl;
    std::cout << "repr: ";
    std::cout << port << "\n";
    std::cout << "ledger" << "\n";
    std::cout << port.ledger() << "\n";
    std::cout << "current prices pre registering data source" << "\n";
    std::cout << port.currentPrices()<< "\n";
    std::cout << "current prices post registering data source" << "\n";
    port.setDataSource(&dataSource);
    std::cout << port.currentPrices()<< "\n";
    assert(port.ledger().size() == port.nAssets());
  }
  portfolio1.setDataSource(&dataSource);
  portfolio2.setDataSource(&dataSource);
  portfolio3.setDataSource(&dataSource);
  std::cout << "current prices" << "\n";
  std::cout << portfolio1.currentPrices() << std::endl;

  assert(dataSource.getData().isApprox(portfolio1.currentPrices()));
  assert(portfolio2.currentPrices().isApprox(portfolio3.currentPrices()));
  assert(portfolio1.cash() == portfolio2.cash());
  assert(portfolio2.cash() == portfolio3.cash());
  assert(portfolio1.equity() == portfolio2.equity());
  assert(portfolio2.equity() == portfolio3.equity());

}

void testAccount(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Portfolio portfolio1 = Portfolio("port_for_acc_init", assets, 1'000'000);
  Account account1 = Account("Account_Test", assets, 1'000'000);
  Account account2 = Account(assets, 1'000'000);
  Account account3 = Account(portfolio1);
  Account account4 = Account(portfolio1);
  account4.addPortfolio(Portfolio("extra_port_added1", assets, 1'000'000));
  account4.addPortfolio(Portfolio("extra_port_added2", assets, 1'000'000));
  account4.addPortfolio(Portfolio("extra_port_added3", assets, 1'000'000));

  Synth dataSource = Synth();
  account1.setDataSource(&dataSource);
  account2.setDataSource(&dataSource);
  account3.setDataSource(&dataSource);
  account4.setDataSource(&dataSource);

  dataSource.getData();
  assert(dataSource.getData().isApprox(account1.currentPrices()));
  assert(account2.currentPrices().isApprox(account3.currentPrices()));

  int i = 0;
  for (auto& acc: {account1, account2, account3}){
    const PortfolioBook portBook= acc.portfolioBook();
    const Portfolio& port=acc.portfolio();
    const DataSource* datSource=acc.dataSource();
    assert(&port == &(acc.defaultPortfolio())); // check pointers to same object
    assert(dataSource.getData().isApprox(port.currentPrices()));
    assert(dataSource.getData().isApprox(datSource->currentData()));
    for(auto it = portBook.begin(); it!= portBook.end(); it++){
      std::cout<<*(it->second)<<"\n";
    }
  }
  for (const auto& port: account4.portfolios()){
    const DataSource* datSource{nullptr};
    datSource=port.dataSource();
    assert(dataSource.getData().isApprox(port.currentPrices()));
    assert(dataSource.getData().isApprox(datSource->currentData()));
  }

  bool caught{false};
  try{
    account1.portfolio("portfolio_doesn't_exist");
  }
  catch (const std::out_of_range& oor){
    std::cerr<< "Expected exception caught: \n";
    std::cerr << "Portfolio doesn't exist: " << oor.what() << "\n";
    caught=true;
  }
  assert(caught);

}

void testBroker(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Account account1("acc01", assets, 1'000'000);
  Broker broker1(assets, 1'000'000);
  Broker broker2("testing account", assets, 1'000'000);
  Broker broker3(account1);
  Broker broker4(account1.portfolio());
  Synth dataSource = Synth();
  DataSource* pdataSource = &dataSource;

  assert(&account1 != &broker3.account("acc01"));
  assert(&account1.portfolio() != &broker3.account("acc01").portfolio());

  dataSource.getData();
  broker1.setDataSource(pdataSource);
  broker2.setDataSource(pdataSource);
  std::cout << broker1.account().portfolio().currentPrices()<< "\n";
  std::cout << broker2.account().portfolio().currentPrices()<< "\n";
  for(const auto pbroker: {&broker1, &broker2}){
    const Broker& broker = *pbroker;
    const PriceVector& sourceRef = dataSource.currentData();
    const PriceVector& brokerRef= broker.currentPrices();
    const PriceVector& accountRef= broker.account().currentPrices();
    const PriceVectorMap& portfolioRef= broker.account().portfolio().currentPrices();
    const Portfolio& portfolio = broker.account().portfolio();
    const  DataSource* datSource=portfolio.dataSource();
    const PriceVector& portfolioRef2=datSource->currentData();
    assert(dataSource.getData().isApprox(brokerRef));
    assert(dataSource.getData().isApprox(accountRef));
    assert(dataSource.getData().isApprox(portfolioRef));
    assert(dataSource.getData().isApprox(portfolioRef2));
  }
}
void testEnv(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // std::unique_ptr<DataSource> dataSource1 = std::make_unique<Synth>();
  Broker broker1 = Broker(assets, 1'000'000);
  Env env = Env("Synth", assets, 1'000'000);
}

void testEnvConfig(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  vector<double> _freq{1., 0.3, 2., 0.5};
  vector<double> _mu{2., 2.1, 2.2, 2.3};
  vector<double> _amp{1., 1.2, 1.3, 1.};
  vector<double> _phase{0., 1., 2., 1.};
  double _dX = 0.01;
  Config config{
    {
      "generator_params", Config{{"freq", _freq},
                           {"mu", _mu},
                           {"amp", _amp},
                           {"phase", _phase},
                           {"dX", _dX}}
    }
  };
  Env env = Env("Synth", assets, 1'000'000, config);
}

void testTransactionHandling(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Synth dataSource = Synth();
  dataSource.getData();
  Portfolio port = Portfolio("port_for_acc_init", assets, 1'000'000);
  Account acc = Account("Account_Test", assets, 1'000'000);
  Broker broker = Broker("BrokerAcc", assets, 1'000'000);
  broker.setDataSource(&dataSource);
  broker.addAccount(acc);
  broker.setRequiredMargin("Account_Test", 0.1);
  port.setDataSource(&dataSource);
  acc.setDataSource(&dataSource);
  acc.addPortfolio("manually_added_port", assets, 1'000'000);
  acc.addPortfolio(port);
  acc.setDefaultPortfolio(acc.portfolios()[0].id());
  double price = dataSource.currentData()[0];
  port.handleTransaction(0, price, 10'000,
                         0., 1.);
  port.handleTransaction("sine1", price, 10'000,
                         0., 1.);
  acc.handleTransaction(0, price, 10'000,
                         0., 1.);
  acc.handleTransaction("sine1", price, 10'000,
                         0., 1.);
  acc.handleTransaction("manually_added_port", 0, price, 10'000,
                        0., 1.);
  acc.handleTransaction("manually_added_port", "sine1", price, 10'000,
                        0., 1.);
  broker.handleTransaction("sine1", 10'000);
  broker.handleTransaction(0, 10'000);
  broker.handleTransaction("BrokerAcc", "sine1", 10'000);
  broker.handleTransaction("BrokerAcc", 0, 10'000);
  broker.handleTransaction("Account_Test", "port_0", "sine1", 10'000);
  broker.handleTransaction("Account_Test", "port_0", 0, 10'000);
  assert(acc.equity() == 3*port.equity());
  assert(acc.cash() == 3*1'000'000 - 4*price*10'000);
  assert(acc.borrowedMargin() == 2*port.borrowedMargin());
  assert(broker.account("Account_Test").cash() == 1'000'000-2*0.1*price*10'000);
  assert(broker.account("Account_Test").borrowedMargin() == 2*0.9*price*10'000);
}

void testAccountingPortfolio(){
  ///////////////////////////////
  // Done via envTest.py - easier
  ///////////////////////////////
  // // Synth dataSource = Synth();
  // Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // // dataSource.getData();
  // // Broker broker = Broker(assets, 1'000'000);
  // std::unique_ptr<DataSource> dataSource1 = std::make_unique<Synth>();
  // // Env env = Env(std::move(dataSource1), assets, 1'000'000);
  // Env env = Env("Synth", assets, 1'000'000);
  // // const PriceVector& sourceRef = env.dataSource()->getData();
  // const PriceVector& envRef = env.currentData();
  // std::cout << "env Ref: ";
  // std::cout << envRef << "\n";
}

int main(){

  std::cout<< "===========================================\n";
  std::cout<< "testDataSource();\n";
  testDataSource();
  std::cout<< "===========================================\n";
  std::cout<< "testPortfolio();\n";
  testPortfolio();
  std::cout<< "===========================================\n";
  std::cout<< "testAccount();\n";
  testAccount();
  std::cout<< "===========================================\n";
  std::cout<< "testBroker();\n";
  testBroker();
  std::cout<< "===========================================\n";
  std::cout<< "testEnv();\n";
  testEnv();
  std::cout<< "===========================================\n";
  std::cout<< "testEnvConfig();\n";
  std::cout<< "===========================================\n";
  testEnvConfig();
  std::cout<< "===========================================\n";
  std::cout<< "testTransactionHandling();\n";
  std::cout<< "===========================================\n";
  testTransactionHandling();
  std::cout<< "===========================================\n";
  std::cout<< "testAccountingportfolio();\n";
  std::cout<< "===========================================\n";
  testAccountingPortfolio();
  std::cout<< "===========================================\n";
  std::cout<< "TESTS COMPLETED\n";
  std::cout<< "===========================================\n";


  return 0;
}

