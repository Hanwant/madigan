#ifndef PORTFOLIO_H_
#define PORTFOLIO_H_

#include <vector>
#include <string>

namespace madigan{

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
  int nAssets;
  double initCash;

 public:
  Portfolio();
 Portfolio(int nAssets, int initCash, std::vector<double> portfolio): nAssets(nAssets),
    initCash(initCash), _portfolio(portfolio){};
  Portfolio(int nAssets, int initCash){
    /* vector<double> port(nAssets, 0.); */
    Portfolio(nAssets, initCash, std::vector<double>(nAssets, 0.));
  };
  ~Portfolio();
  double cash(){return _cash;};
  std::vector<double> portfolio();
  std::vector<double> portfolioNormed();
  double equity();
  double availableMargin();
  double borrowedCash();


 private:
  std::vector<double> _portfolio;
  double _cash;
};


} /*namespace madigan*/

#endif /*  PORTFOLIO_H_ */
