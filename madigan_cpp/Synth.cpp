#include "Synth.h"


std::vector<double> Synth::generate(){
  return generator->next();
}

State Synth::preprocess(std::vector<double> prices){
  std::vector<double> normedPortfolio = portfolioNorm();

  return State(prices, normedPortfolio);
}

double Synth::equity(){
  double _sum(0.);
  for (int i=0; i != nAssets; i++){
    _sum += (_currentPrices[i] * _portfolio[i]);
  }
  return _cash + _sum;
}

double Synth::availableMargin(){
  double _sum(0.);
  for (int i=0; i != nAssets; i++){
    double port = _portfolio[i];
    if(port < 0.){
      _sum += (_currentPrices[i] * port);
    }
  }
  return _cash + _sum;
}

bool Synth::checkRisk(int assetId, double amount){
  if (amount == 0.) return false;
  if (amount > 0.){
    if (_portfolio[assetId] < 0.) return true;
    if(amount < availableMargin()) return true;
  }
  else{
    if (_portfolio[assetId] > 0.)return true;
    if (abs(amount) < availableMargin()) return true;
  }
}

void Synth::transaction(int assetId, double amount, double transPrice, double transCost){
  double unitsToExchange = amount / transPrice;
  _cash -= (amount + transCost);
  _portfolio[assetId] += unitsToExchange;
}

envOutput Synth::step(std::vector<int> actions){
  if (actions.size() != nAssets){
    throw std::length_error("Actions must be of same length as number of assets");
  }

  if (equity() <= 0.){
    bool done = true;
    double reward = 0.;
    Info info("BLOWNOUT",
              std::vector<double>(nAssets, 0.),
              std::vector<double>(nAssets, 0.));
    std::vector<double> data = generate();
    _currentPrices = data;
    State state(_currentPrices, portfolioNorm());
    return std::make_tuple(state, reward, done, info);
  }
  if (availableMargin() <= _maintenanceMargin){
    bool done = true;
    double reward = 0.;
    Info info("MARGINCALL",
              std::vector<double>(nAssets, 0.),
              std::vector<double>(nAssets, 0.));
    std::vector<double> data = generate();
    _currentPrices = data;
    State state(_currentPrices, portfolioNorm());
    return std::make_tuple(state, reward, done, info);
  }

  double currentEquity = equity();
  Info info;

  if (discreteActions){
    info = stepDiscrete(actions);
  }
  else{
    throw "Non Discrete Action not yet implemented";
      }

  _currentPrices = generate();

  double nextEquity = equity();
  double immediateReturn = (nextEquity/currentEquity) - 1.;

  State state = preprocess(_currentPrices);

  envOutput out=std::make_tuple(state, immediateReturn, false, info);
  return out;

}


Info Synth::stepDiscrete(std::vector<int> actions){
  std::vector<double> transPrices;
  std::vector<double> transCosts;
  for(int i=0; i != nAssets; i++){
    double price = _currentPrices[i];
    int lot =  lotUnitValue * (actions[i] - discreteActionAtomsShift);
    double amount = lot; //* transPrice;
    double slippage = price * slippagePct * (lot<0? -1: 1);
    double transPrice = price + slippage;
    if (checkRisk(i, amount)){
      transaction(i, amount, transPrice, transactionCost);
      transPrices.push_back(transPrice);
      transCosts.push_back(transactionCost);
    }
    else{
      transPrices.push_back(0.);
      transCosts.push_back(0.);
    }
  }
  Info info(transPrices, transCosts);
  return info;
}

