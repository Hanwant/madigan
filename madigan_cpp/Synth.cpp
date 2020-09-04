#include "Synth.h"


std::vector<float> Synth::generate(){
  return generator->next();
}

State Synth::preprocess(std::vector<float> prices){
  std::vector<float> normedPortfolio = portfolioNorm();

  return State(prices, normedPortfolio);
}

float Synth::equity(){
  float _sum(0.);
  for (int i=0; i != nAssets; i++){
    _sum += (_currentPrices[i] * _portfolio[i]);
  }
  return _cash + _sum;
}

float Synth::availableMargin(){
  float _sum(0.);
  for (int i=0; i != nAssets; i++){
    float port = _portfolio[i];
    if(port < 0.){
      _sum += (_currentPrices[i] * port);
    }
  }
  return _cash + _sum;
}

bool Synth::checkRisk(int assetId, float amount){
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

void Synth::transaction(int assetId, float amount, float transPrice, float transCost){
  float unitsToExchange = amount / transPrice;
  _cash -= (amount + transCost);
  _portfolio[assetId] += unitsToExchange;
}

envOutput Synth::step(std::vector<int> actions){
  if (actions.size() != nAssets){
    throw std::length_error("Actions must be of same length as number of assets");
  }

  if (equity() <= 0.){
    bool done = true;
    float reward = 0.;
    Info info("BLOWNOUT",
              std::vector<float>(nAssets, 0.),
              std::vector<float>(nAssets, 0.));
    std::vector<float> data = generate();
    _currentPrices = data;
    State state(_currentPrices, portfolioNorm());
    return std::make_tuple(state, reward, done, info);
  }
  if (availableMargin() <= _maintenanceMargin){
    bool done = true;
    float reward = 0.;
    Info info("MARGINCALL",
              std::vector<float>(nAssets, 0.),
              std::vector<float>(nAssets, 0.));
    std::vector<float> data = generate();
    _currentPrices = data;
    State state(_currentPrices, portfolioNorm());
    return std::make_tuple(state, reward, done, info);
  }

  float currentEquity = equity();
  Info info;

  if (discreteActions){
    info = stepDiscrete(actions);
  }
  else{
    throw "Non Discrete Action not yet implemented";
      }

  _currentPrices = generate();

  float nextEquity = equity();
  float immediateReturn = (nextEquity/currentEquity) - 1.;

  State state = preprocess(_currentPrices);

  envOutput out=std::make_tuple(state, immediateReturn, false, info);
  return out;

}


Info Synth::stepDiscrete(std::vector<int> actions){
  std::vector<float> transPrices;
  std::vector<float> transCosts;
  for(int i=0; i != nAssets; i++){
    float price = _currentPrices[i];
    int lot =  lotUnitValue * (actions[i] - discreteActionAtomsShift);
    float slippage = price * slippagePct * (lot<0? -1: 1);
    float transPrice = price + slippage;
    float amount = lot * transPrice;
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

