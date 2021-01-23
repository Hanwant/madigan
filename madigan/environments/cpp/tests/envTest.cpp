#include <iostream>
#include <assert.h>
#include <stdexcept>
#include <iomanip>

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
    const DataSourceTick* datSource=acc.dataSource();
    assert(&port == &(acc.defaultPortfolio())); // check pointers to same object
    assert(dataSource.getData().isApprox(port.currentPrices()));
    assert(dataSource.getData().isApprox(datSource->currentData()));
    for(auto it = portBook.begin(); it!= portBook.end(); it++){
      std::cout<<*(it->second)<<"\n";
    }
  }
  for (const auto& port: account4.portfolios()){
    const DataSourceTick* datSource{nullptr};
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
  DataSourceTick* pdataSource = &dataSource;

  assert(&account1 != &broker3.account("acc01"));
  assert(&account1.portfolio() != &broker3.account("acc01").portfolio());

  dataSource.getData();
  broker1.setDataSource(pdataSource);
  broker2.setDataSource(pdataSource);
  std::cout << broker1.account().portfolio().currentPrices()<< "\n";
  std::cout << broker2.account().portfolio().currentPrices()<< "\n";
  for(const auto& pbroker: {&broker1, &broker2}){
    const Broker& broker = *pbroker;
    const PriceVector& sourceRef = dataSource.currentData();
    const PriceVectorMap& brokerRef= broker.currentPrices();
    const PriceVectorMap& accountRef= broker.account().currentPrices();
    const PriceVectorMap& portfolioRef= broker.account().portfolio().currentPrices();
    const Portfolio& portfolio = broker.account().portfolio();
    const  DataSourceTick* datSource=portfolio.dataSource();
    const PriceVector& portfolioRef2=datSource->currentData();
    assert(dataSource.getData().isApprox(brokerRef));
    assert(dataSource.getData().isApprox(accountRef));
    assert(dataSource.getData().isApprox(portfolioRef));
    assert(dataSource.getData().isApprox(portfolioRef2));
  }

  // 2 types of broker response
  // BrokerResponseSingle for single transactions
  // BrokerResponseMulti for multiple sent/executed transactions
  BrokerResponseSingle brokerResp1(0.1, 10., 0.1, RiskInfo::green);
  PriceVector transPrices(1);
  AmountVector transUnits(1);
  PriceVector transCosts(1);
  transPrices << 1.;
  transUnits << 1.;
  transCosts << 1.;
  BrokerResponseMulti brokerResp2(transPrices, transUnits, transCosts,
                                  std::vector<RiskInfo>{RiskInfo::green});


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
                         0.);
  port.handleTransaction("sine1", price, 10'000,
                         0.);
  acc.handleTransaction(0, price, 10'000,
                         0.);
  acc.handleTransaction("sine1", price, 10'000,
                         0.);
  acc.handleTransaction("manually_added_port", 0, price, 10'000,
                        0.);
  acc.handleTransaction("manually_added_port", "sine1", price, 10'000,
                        0.);
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
  // Done mostly  via envTest.py - easier
  ///////////////////////////////
  // Synth dataSource = Synth();
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  Synth dataSource = Synth();
  PriceVector oldPrices = dataSource.getData();
  Portfolio port = Portfolio("port_for_acc_init", assets, 1'000'000);
  port.setDataSource(&dataSource);
  port.handleTransaction(0, oldPrices(0), 10'000,0.);
  assert(port.pnl() == 0.);
  PriceVector newPrices = dataSource.getData();
  assert(!oldPrices.isApprox(newPrices));
  double manualPNL = 10'000*(newPrices(0)-oldPrices(0));
  assert(0.000001 > (port.pnl() - manualPNL));
  std::cout <<std::setprecision(20)<< port.pnl() << "\n";
  std::cout << std::setprecision(20)<< manualPNL << "\n";
  assert(abs(port.pnl() - manualPNL) < 0.0000001);
  std::cout << "equity: " << port.equity() << "\n";
  double diff1 = abs(port.equity() - (port.cash() + port.assetValue() - port.borrowedMargin()));
  double diff2 = abs(port.equity() - (port.balance() + port.pnl() + port.usedMargin()));
  double diff3 = abs(port.equity() - (port.cash() + port.borrowedAssetValue() +
                                      port.pnl() + port.usedMargin()));
  // std::cout <<std::setprecision(20)<< port.usedMargin()<< "\n";
  // std::cout <<std::setprecision(20)<< port.borrowedMargin()<< "\n";
  // std::cout <<std::setprecision(20)<< port.assetValue()<< "\n";
  // std::cout <<std::setprecision(20)<< port.borrowedAssetValue()<< "\n";
  // std::cout <<std::setprecision(20)<< port.borrowedEquity()<< "\n";
  // std::cout <<std::setprecision(20)<< port.ledger()<< "\n";
  // std::cout <<std::setprecision(20)<< port.currentPrices()<< "\n";
  std::cout <<std::setprecision(20)<< diff1 << "\n";
  std::cout <<std::setprecision(20)<< diff2 << "\n";
  std::cout <<std::setprecision(20)<< diff3 << "\n";
  assert(diff1 < 0.0000000001);
  assert(diff2 < 0.0000000001);
  assert(diff3 < 0.0000000001);
  port.handleTransaction(0, newPrices(0), 10'000,0.);
  port.handleTransaction(0, newPrices(0), -20'000,0.);
  std::cout <<std::setprecision(20)<< port.pnl() << "\n";
  assert(port.pnl() == 0.);
  port.handleTransaction(0, newPrices(0), -10'000,0.);
  std::cout << "Asset Value\n";
  std::cout <<std::setprecision(20)<< port.assetValue() << "\n";
  std::cout << "mean entry price\n";
  std::cout <<std::setprecision(20)<< port.meanEntryPrices() << "\n";
  std::cout << "ledger\n";
  std::cout <<std::setprecision(20)<< port.ledger() << "\n";
  std::cout << "mean entry value\n";
  std::cout <<std::setprecision(20)<< port.meanEntryValue() << "\n";
  std::cout << "pnl\n";
  std::cout <<std::setprecision(20)<< port.pnl() << "\n";
  assert(port.pnl() == 0.);
  port.close(0, newPrices(0), 0.);
}

void testEnvInit(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // std::unique_ptr<DataSource> dataSource1 = std::make_unique<Synth>();
  Broker broker1 = Broker(assets, 1'000'000);
  Env env = Env("Synth", 1'000'000);
  SRDI<double> envResponse = env.step();
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
      "data_source_config", Config{{"freq", _freq},
                                 {"mu", _mu},
                                 {"amp", _amp},
                                 {"phase", _phase},
                                 {"dX", _dX}}
    }
  };
  Env env = Env("Synth", 1'000'000, config);
}

void testEnvData(){
  Assets assets{"sine1", "sine2", "sine3", "sine4"};
  // std::unique_ptr<DataSource> dataSource1 = std::make_unique<Synth>();
  Broker broker1 = Broker(assets, 1'000'000);
  Env env = Env("Synth",  1'000'000);
  SRDI<double> envResponse = env.step();
  std::cout << env.currentPrices() << "\n";
}

template<class T>  // T expected to be an EigenMatrix
T loadMatrixFromHDF(string fname, string groupKey, string vectorKey,
                    std::size_t start=0, std::size_t  size=-1){
  H5Easy::File file(fname, HighFive::File::ReadOnly);
  string datasetPath = "/"+groupKey+"/"+vectorKey;
  vector<size_t> shape = H5Easy::getShape(file, datasetPath);
  // T out(len);
  // out = H5Easy::load<T>(file, datasetPath);
  if (size == -1) size = shape[0];
  T out(size, shape[1]);
  HighFive::DataSet dataset = file.getDataSet(datasetPath);
  dataset.select({start, 0}, {size, shape[1]}).read(out.data());
  return out;
}

void testHDFSourceSingle(){
  using namespace HighFive;

  string filepath="test_envTest.h5";
  string mainKey="group/dataset";
  string priceKey="midprice";
  string featureKey="feats";
  string timestampKey="timestamp";
  vector<string> assets{"Test"};
  {
  File file(filepath, File::ReadWrite | File::Create | File::Truncate);
  // Eigen::VectorXd price (10);
  // Eigen::Vector<int, Eigen::Dynamic> timestamps (10);
  vector<double> price(10);
  vector<vector<double>> feats(10, vector<double>(1, 0.));
  vector<std::uint64_t> timestamps(10);
  for (int i=0; i<10; i++){
    price[i] = (double)i;
    feats[i][0] = (double)i*i*i;
    timestamps[i] = i * i;
  }
  std::vector<std::size_t> featDims {10, 1};
  std::vector<std::size_t> vecDims {10};
  Group group = file.createGroup(mainKey);
  Attribute attr = group.createAttribute("assets", assets);
  attr.write(assets);

  DataSet priceDataset = file.createDataSet<double>(mainKey+'/'+priceKey,
                                                    DataSpace(vecDims));
  DataSet featsDataset = file.createDataSet<double>(mainKey+'/'+featureKey,
                                                    DataSpace(featDims));
  DataSet timestampsDataset = file.createDataSet<std::uint64_t>(mainKey+'/'+timestampKey,
                                                         DataSpace(vecDims));
  priceDataset.write(price);
  featsDataset.write(feats);
  timestampsDataset.write(timestamps);
  // H5Easy Works too
  // H5Easy::dump(file, mainKey+'/'+priceKey, price);
  // H5Easy::dump(file, mainKey+'/'+timestampKey, timestamps);
  // {
  //   PriceMatrix data;
  //   DataSet dset = file.getDataSet(mainKey+'/'+priceKey);
  //   vector<size_t> shape = H5Easy::getShape(file, mainKey+'/'+priceKey);
  //   data.resize(shape[0], shape[1]);
  //   dset.select({0, 0}, {5, 1}).read(data.data());
  //   std::cout << data << "\n";
  //   std::cout <<"done explicit load\n";
  // }

  }

  HDFSourceSingle dataSource(filepath, mainKey, priceKey, featureKey,
                             timestampKey, 10);
  dataSource.getData(); dataSource.getData(); dataSource.getData();
  std::cout << "Prices: " << dataSource.currentPrices() << "\n";
  std::cout << "Feats: " << dataSource.currentData() << "\n";
  std::cout << "Assets: " << dataSource.assets() << "\n";
  std::cout << "Time Bounds: " << dataSource.startTime << ", " << dataSource.endTime << "\n";
  std::cout << "Bounds: " << dataSource.boundsIdx().first << ", "
            << dataSource.boundsIdx().second << "\n";

  // Check Specification of time range - should end up with boundsIdx (1, 8)
  HDFSourceSingle dataSource2(filepath, mainKey, priceKey, featureKey,
                              timestampKey, 10, 1, 63);

  dataSource2.getData(); dataSource2.getData(); dataSource2.getData();
  std::cout << "Prices: " << dataSource2.currentPrices() << "\n";
  std::cout << "Feats: " << dataSource2.currentData() << "\n";
  std::cout << "Assets: " << dataSource2.assets() << "\n";
  std::cout << "Time Bounds: " << dataSource2.startTime << ", " << dataSource2.endTime << "\n";
  std::cout << "Bounds: " << dataSource2.boundsIdx().first << ", "
            << dataSource2.boundsIdx().second << "\n";

  assert(dataSource2.boundsIdx().first == 1);
  assert(dataSource2.boundsIdx().second == 8);

  // If startTime or endTime is outside of range of dataset
  // raise exception
  bool caught = false;
  try{
    HDFSourceSingle dataSource3(filepath, mainKey, priceKey, featureKey,
                                timestampKey, 10, 0, 82);
  } catch (const std::out_of_range& oor){
    std::cerr<< "Expected exception caught: \n";
    std::cerr << oor.what() << "\n";
    caught=true;
  }
  assert(caught);
}

// void testHDFSourceMulti(){
//   using namespace HighFive;

//   string filepath="test_envTest.h5";
//   string mainKey="group/dataset";
//   string priceKey="midprice";
//   string timestampKey="timestamp";
//   File file(filepath, File::ReadWrite | File::Create | File::Truncate);
//   // Eigen::VectorXd price (10);
//   // Eigen::Vector<int, Eigen::Dynamic> timestamps (10);
//   vector<double> price(10);
//   vector<int> timestamps(10);
//   for (int i=0; i<10; i++){
//     price[i] = (double)i;
//     timestamps[i] = i;
//   }
//   std::vector<std::size_t> Dims {10};
//   file.createGroup(mainKey);
//   DataSet priceDataset = file.createDataSet<double>(mainKey+'/'+priceKey, DataSpace(Dims));
//   DataSet timestampsDataset = file.createDataSet<int>(mainKey+'/'+timestampKey, DataSpace(Dims));
//   // H5Easy::dump works without having to createGroup
//   // H5Easy::dump(file, mainKey+'/'+priceKey, price);
//   // H5Easy::dump(file, mainKey+'/'+timestampKey, timestamps);
//   priceDataset.write(price);
//   timestampsDataset.write(timestamps);

//   HDFSource dataSource(filepath, mainKey, priceKey, timestampKey);
//   std::cout << dataSource.getData() << "\n";
// }



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
  std::cout<< "testTransactionHandling();\n";
  testTransactionHandling();
  std::cout<< "===========================================\n";
  std::cout<< "testAccountingportfolio();\n";
  testAccountingPortfolio();
  std::cout<< "===========================================\n";
  std::cout<< "testEnvInit();\n";
  testEnvInit();
  std::cout<< "===========================================\n";
  std::cout<< "testEnvConfig();\n";
  testEnvConfig();
  std::cout<< "===========================================\n";
  std::cout<< "testEnvData();\n";
  testEnvData();
  std::cout<< "===========================================\n";
  std::cout<< "testHDFSourceSingle();\n";
  testHDFSourceSingle();
  // std::cout<< "===========================================\n";
  // std::cout<< "testHDFSourceMulti();\n";
  // testHDFSourceMulti();
  std::cout<< "===========================================\n";
  std::cout<< "TESTS COMPLETED\n";


  return 0;
}

