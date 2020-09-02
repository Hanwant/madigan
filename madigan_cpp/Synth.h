#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <torch/torch.h>
#include "Generator.cpp"

using std::string;

/* typedef torch::Tensor State; */

template<typename T>
struct _State{
  T price;
  T port;
_State(T price, T port): price(price), port(port){};
};

struct Info{
  string event;
  std::vector<float> transactionPrices;
  std::vector<float> transactionCosts;

Info(string event, std::vector<float> transPrices, std::vector<float> transCosts):
  event(event), transactionPrices(transPrices), transactionCosts(transCosts){}

Info(std::vector<float> transPrices, std::vector<float> transCosts):
  event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
  Info(){};

};

typedef _State<std::vector<float>> State;
typedef std::tuple<State, float, bool, Info> envOutput;


class Synth{
public:
  int nAssets;
  int minTf;
  float transactionCost;
  float slippagePct;
  bool discreteActions=true;
  int discreteActionAtoms;
  int lotUnitValue;

public:
  Synth(): nAssets(4), minTf(64), _initCash(1'000'000),
    discreteActionAtoms(11), lotUnitValue(1'000)
    {
      std::cout << "Default constructor" << std::endl;
      discreteActionAtomsShift = discreteActionAtoms / 2;
    }

 Synth(int nAssets, int minTf, int initCash,
       int discreteActionAtoms=11, int lotUnitValue=1'000,
       float transactionCost=0.01, float slippagePct=0.001):
  nAssets(nAssets), minTf(minTf), _initCash(initCash),
    discreteActionAtoms(discreteActionAtoms), lotUnitValue(lotUnitValue),
    transactionCost(transactionCost), slippagePct(slippagePct)
    {
      std::cout << "Specified Constructor" << std::endl;
      discreteActionAtomsShift = discreteActionAtoms / 2;
    }
  envOutput step(std::vector<int> actions);
  int cash(){ return _cash;};
  float equity();
  float availableMargin();
  std::vector<float> portfolio(){ return _portfolio;};
  std::vector<float> portfolioNorm(){ return _portfolio;};
  std::vector<float> currentPrices(){ return _currentPrices;};

 private:
  std::vector<float> generate();
  State preprocess(std::vector<float>);
  bool checkRisk(int assetId, int amount);
  Info stepDiscrete(std::vector<int> actions);
  Info stepContinuous(std::vector<int> actions);
  void transaction(int assetId, float amount, float transPrice, float transCost);


private:
  int _cash;
  int _initCash;
  int discreteActionAtomsShift;
  std::vector<float> _portfolio;
  std::vector<float> _currentPrices;
  float _maintenanceMargin = 0.1;
  Generator generator = SineGenerator();
};
