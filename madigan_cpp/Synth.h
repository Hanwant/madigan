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
  std::vector<double> transactionPrices;
  std::vector<double> transactionCosts;

Info(string event, std::vector<double> transPrices, std::vector<double> transCosts):
  event(event), transactionPrices(transPrices), transactionCosts(transCosts){}

Info(std::vector<double> transPrices, std::vector<double> transCosts):
  event(""), transactionPrices(transPrices), transactionCosts(transCosts){}
  Info(){};

};

typedef _State<std::vector<double>> State;
typedef std::tuple<State, double, bool, Info> envOutput;

class Synth{
public:
  int nAssets;
  int minTf;
  double transactionCost;
  double slippagePct;
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
       double transactionCost=0.01, double slippagePct=0.001):
  nAssets(nAssets), minTf(minTf), _initCash(initCash),
    discreteActionAtoms(discreteActionAtoms), lotUnitValue(lotUnitValue),
    transactionCost(transactionCost), slippagePct(slippagePct)
    {
      std::cout << "Specified Constructor" << std::endl;
      discreteActionAtomsShift = discreteActionAtoms / 2;
    }
  envOutput step(std::vector<int> actions);
  int cash(){ return _cash;};
  double equity();
  double availableMargin();
  std::vector<double> portfolio(){ return _portfolio;}
  std::vector<double> portfolioNorm(){ return _portfolio;}
  std::vector<double> currentPrices(){ return _currentPrices;}
  std::size_t currentTimestamp() const { return _currentTimestamp; }

 private:
  std::vector<double> generate();
  State preprocess(std::vector<double>);
  bool checkRisk(int assetId, double amount);
  Info stepDiscrete(std::vector<int> actions);
  Info stepContinuous(std::vector<int> actions);
  void transaction(int assetId, double amount, double transPrice, double transCost);


private:
  int _cash;
  int _initCash;
  int discreteActionAtomsShift;
  std::vector<double> _portfolio;
  std::vector<double> _currentPrices;
  std::size_t _currentTimestamp{0};
  double _maintenanceMargin = 0.1;
  std::unique_ptr<Generator> generator = std::make_unique<SineGenerator>();
};

