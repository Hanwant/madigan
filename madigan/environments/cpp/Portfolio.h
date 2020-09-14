#ifndef PORTFOLIO_H_
#define PORTFOLIO_H_

#include <vector>
#include <string>

#include "Assets.h"


namespace madigan{

using std::string;

struct Info{
  std::string event;
  std::vector<double> transactionPrices;
  std::vector<double> transactionCosts;

Info(std::string event, std::vector<double> transPrices, std::vector<double> transCosts):
  event(event), transactionPrices(transPrices), transactionCosts(transCosts){}

Info(std::vector<double> transPrices, std::vector<double> transCosts):
  event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
  Info(){};

};

class Portfolio {
 public:
  Assets assets;
  double initCash;

 public:
  Portfolio(){};
 Portfolio(Assets assets, double initCash): ID("port_default"), assets(assets),
    initCash(initCash){};
 Portfolio(string id, Assets assets, double initCash): ID(id), assets(assets),
    initCash(initCash){
    this->_portfolio = std::vector<double>(assets.size(), 0.);
  };
 Portfolio(string id, Assets assets, double initCash, std::vector<double> portfolio): ID(id), assets(assets),
    initCash(initCash), _portfolio(portfolio){};
 Portfolio(string id, std::vector<string> assets, double initCash): ID(id), assets(assets),
    initCash(initCash){
    this->_portfolio = std::vector<double>(assets.size(), 0.);
  };
 Portfolio(string id, std::vector<string> assets, double initCash, std::vector<double> portfolio): ID(id), assets(assets),
    initCash(initCash), _portfolio(portfolio){};
  ~Portfolio()=default;
  int nAssets(){ return assets.size();};
  string id(){ return ID;};
  double cash(){ return _cash;};
  std::vector<double> portfolio(){return _portfolio;};
  std::vector<double> portfolioNormed();
  double equity();
  double availableMargin();
  double borrowedCash();

  friend class Broker;

 private:
  string ID;
  std::vector<double> _portfolio;
  double _cash;
};


} /*namespace madigan*/

#endif /*  PORTFOLIO_H_ */
