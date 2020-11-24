#include <stdexcept>
#include <iostream>
#include <numeric>

#include "DataSource.h"

namespace madigan{

  std::unique_ptr<DataSource> makeDataSource(string dataSourceType){
    if(dataSourceType == "Synth"){
      return make_unique<Synth>();
    }
    else if (dataSourceType == "SawTooth"){
      return make_unique<SawTooth>();
    }
    else if (dataSourceType == "Triangle"){
      return make_unique<Triangle>();
    }
    else if (dataSourceType == "SineAdder"){
      return make_unique<SineAdder>();
    }
    else if (dataSourceType == "OU"){
      return make_unique<OU>();
    }
    else if (dataSourceType == "SimpleTrend"){
      return make_unique<SimpleTrend>();
    }
    else if (dataSourceType == "TrendOU"){
      return make_unique<TrendOU>();
    }
    else{
      std::stringstream ss;
      ss << "Default Constructor for ";
      ss << dataSourceType;
      ss << " as dataSource is not implemented";
      throw NotImplemented(ss.str());
    }
  }
  std::unique_ptr<DataSource> makeDataSource(string dataSourceType, Config config){
    if(dataSourceType == "Synth"){
      return make_unique<Synth>(config);
    }
    else if (dataSourceType == "SawTooth"){
      return make_unique<SawTooth>(config);
    }
    else if (dataSourceType == "Triangle"){
      return make_unique<Triangle>(config);
    }
    else if (dataSourceType == "SineAdder"){
      return make_unique<SineAdder>(config);
    }
    else if (dataSourceType == "OU"){
      return make_unique<OU>(config);
    }
    else if (dataSourceType == "SimpleTrend"){
      return make_unique<SimpleTrend>(config);
    }
    else if (dataSourceType == "TrendOU"){
      return make_unique<TrendOU>(config);
    }
    else if (dataSourceType == "Composite"){
      return make_unique<Composite>(config);
    }
    else{
      std::stringstream ss;
      ss << "Constructor from config for";
      ss << dataSourceType;
      ss << " as dataSource is not implemented";
      throw NotImplemented(ss.str());
    }
  }

  Composite::Composite(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto singleSource: params){
      string dataSourceType = std::any_cast<string>(singleSource.first);
      Config config = std::any_cast<Config>(singleSource.second);
      std::unique_ptr<DataSource> source = makeDataSource(dataSourceType, config);
      dataSources_.emplace_back(std::move(source));
    }
    for(const auto& source: dataSources_){
      for (const auto& asset: source->assets){
        assets.emplace_back(asset);
      }
      nAssets_ += source->nAssets();
    }
    currentData_.resize(nAssets_);
    // currentPrices_.resize(nAssets_);
  }

  const PriceVector& Composite::getData(){
    int idx{0};
    for (int i=0; i<dataSources_.size(); i++){
      DataSource* source = dataSources_[i].get();
      currentData_.segment(idx, source->nAssets()) = source->getData();
      idx += source->nAssets();
    }
    timestamp_ += 1;
    return currentData_;
  }


  void Synth::initParams(std::vector<double> _freq, std::vector<double> _mu,
                         std::vector<double> _amp, std::vector<double> _phase,
                         double _dX, double _noise)
  {
    this->freq=_freq;
    this->mu=_mu;
    this->amp=_amp;
    this->initPhase=_phase;
    this->x=_phase;
    this->dX=_dX;
    this->noise=_noise;
    this->noiseDistribution = std::normal_distribution<double>(0., _noise);
    for (int i=0; i < freq.size(); i++){
      std::string assetName = "sine_" + std::to_string(i);
      this->assets.push_back(Asset(assetName));
    }
    this->nAssets_ = assets.size();
    currentData_.resize(nAssets_);;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  Synth::Synth(){
    double _dX{0.01};
    vector<double> freq{1., 0.3, 2., 0.5};
    vector<double> mu{2., 2.1, 2.2, 2.3};
    vector<double> amp{1., 1.2, 1.3, 1.};
    vector<double> phase{0., 1., 2., 1.};
    initParams(freq, mu, amp, phase, dX, 0.);
  }

  Synth::Synth(std::vector<double> _freq, std::vector<double> _mu,
               std::vector<double> _amp, std::vector<double> _phase,
               double _dX, double noise){
    std::cout <<"INSIDE SYNTH CALLED W NOISE\n";
    if ((_freq.size() == _mu.size()) && (_mu.size() == _amp.size()) &&
        (_amp.size() == _phase.size())){
      initParams(_freq, _mu, _amp, _phase, _dX, noise);
    }
    else{
      throw std::length_error("parameters passed to DataSource of type Synth"
                              " need to be vectors of same length");
    }
  }

  Synth::Synth(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"freq", "mu", "amp", "phase", "dX"}){
      if (params.find(key) == params.end()){
        allParamsPresent=false;
        throw ConfigError("generator parameters don't have all required constructor arguments");
      }
    }
    if (allParamsPresent){
      vector<double> freq = std::any_cast<vector<double>>(params["freq"]);
      vector<double> mu = std::any_cast<vector<double>>(params["mu"]);
      vector<double> amp = std::any_cast<vector<double>>(params["amp"]);
      vector<double> phase = std::any_cast<vector<double>>(params["phase"]);
      double dX = std::any_cast<double>(params["dX"]);
      if (params.find("noise") != params.end()){
        noise = std::any_cast<double>(params["noise"]);
        initParams(freq, mu, amp, phase, dX, noise);
      }
      else {
        initParams(freq, mu, amp, phase, dX, 0.);
      }

    }
    else{
      vector<double> freq{1., 0.3, 2., 0.5};
      vector<double> mu{2., 2.1, 2.2, 2.3};
      vector<double> amp{1., 1.2, 1.3, 1.};
      vector<double> phase{0., 1., 2., 1.};
      initParams(freq, mu, amp, phase, 0.01, 0.);
    }
  }

  Synth::Synth(pybind11::dict py_config): Synth::Synth(makeConfigFromPyDict(py_config)){
    // Config config = makeConfigFromPyDict(py_config);

  }

  const PriceVector& Synth::getData() {
    for (int i=0; i < nAssets_; i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }

  const pybind11::array_t<double> Synth::getData_np(){
    for (int i=0; i < nAssets_; i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    return pybind11::array_t<double>(
                               {currentData_.size()},
                               {sizeof(double)},
                               currentData_.data()
                               );
  }

  const PriceVector& SawTooth::getData() {
    double intPart;
    for (int i=0; i < nAssets_; i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        amp[i] * std::modf(x[i]*freq[i], &intPart);
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }

  const PriceVector& Triangle::getData() {
    double intPart;
    for (int i=0; i < nAssets_; i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        4*amp[i]/PI2 * std::asin(std::sin(PI2*x[i]/freq[i]));
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }

  // SINE ADDER
  SineAdder::SineAdder(){
    double _dX{0.01};
    vector<double> freq{1., 0.3, 2., 0.5};
    vector<double> mu{2., 2.1, 2.2, 2.3};
    vector<double> amp{1., 1.2, 1.3, 1.};
    vector<double> phase{0., 1., 2., 1.};
    initParams(freq, mu, amp, phase, dX, 0.);
  }

  SineAdder::SineAdder(std::vector<double> _freq, std::vector<double> _mu,
                       std::vector<double> _amp, std::vector<double> _phase,
                       double _dX, double noise){
    if ((_freq.size() == _mu.size()) && (_mu.size() == _amp.size()) &&
        (_amp.size() == _phase.size())){
      initParams(_freq, _mu, _amp, _phase, _dX, noise);
    }
    else{
      throw std::length_error("parameters passed to DataSource of type SineAdder"
                              " need to be vectors of same length");
    }
  }

  SineAdder::SineAdder(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"freq", "mu", "amp", "phase", "dX"}){
      if (params.find(key) == params.end()){
        allParamsPresent=false;
        throw ConfigError("generator parameters don't have all required constructor arguments");
      }
    }
    if (allParamsPresent){
      vector<double> freq = std::any_cast<vector<double>>(params["freq"]);
      vector<double> mu = std::any_cast<vector<double>>(params["mu"]);
      vector<double> amp = std::any_cast<vector<double>>(params["amp"]);
      vector<double> phase = std::any_cast<vector<double>>(params["phase"]);
      double dX = std::any_cast<double>(params["dX"]);
      if (params.find("noise") != params.end()){
        noise = std::any_cast<double>(params["noise"]);
        initParams(freq, mu, amp, phase, dX, noise);
      }
      else {
        initParams(freq, mu, amp, phase, dX, 0.);
      }

    }
    else{
      vector<double> freq{1., 0.3, 2., 0.5};
      vector<double> mu{2., 2.1, 2.2, 2.3};
      vector<double> amp{1., 1.2, 1.3, 1.};
      vector<double> phase{0., 1., 2., 1.};
      initParams(freq, mu, amp, phase, 0.01, 0.);
    }
  }

  SineAdder::SineAdder(pybind11::dict py_config): SineAdder::SineAdder(makeConfigFromPyDict(py_config)){
    // Config config = makeConfigFromPyDict(py_config);

  }
  void SineAdder::initParams(std::vector<double> _freq, std::vector<double> _mu,
                             std::vector<double> _amp, std::vector<double> _phase,
                             double _dX, double _noise)
  {
    this->freq=_freq;
    this->mu=_mu;
    this->amp=_amp;
    this->initPhase=_phase;
    this->x=_phase;
    this->dX=_dX;
    this->noise=_noise;
    this->noiseDistribution = std::normal_distribution<double>(0., _noise);
    this->assets = vector<Asset>(1, Asset("multi_sine"));
    this->nAssets_ = this->assets.size();
    currentData_.resize(1);
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  const PriceVector& SineAdder::getData() {
    double sum{0.};
    for (int i=0; i < freq.size(); i++){
      sum += noiseDistribution(generator) + mu[i] +
        amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    currentData_[0] = sum;
    timestamp_ += 1;
    return currentData_;
  }

  void OU::initParams(std::vector<double> mean, std::vector<double> theta,
                      std::vector<double> phi, std::vector<double> noise_var ) {
    if ((mean.size() == theta.size()) && (theta.size() == phi.size())
        && (phi.size() == noise_var.size())){
      this->mean=mean;
      this->theta=theta;
      this->phi=phi;
      this->noise_var=noise_var;
      for (auto noise: noise_var){
        noiseDistribution.push_back(std::normal_distribution<double>(0., noise));
      }
      for (int i=0; i < mean.size(); i++){
        std::string assetName = "OU_" + std::to_string(i);
        this->assets.push_back(Asset(assetName));
      }
      this->nAssets_ = assets.size();
      currentData_.resize(assets.size());
      for (const auto& val: mean){
        currentData_ << val;
      }
      generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    else{
      throw std::length_error("parameters passed to DataSource of type OU"
                              " need to be vectors of same length");
    }
  }

  OU::OU(std::vector<double> mean, std::vector<double> theta, std::vector<double> phi,
              std::vector<double> noise_var ) {
    initParams(mean, theta, phi, noise_var);
  }
  OU::OU(): OU({2., 4.3, 3., 0.5}, {1., 0.3, 2., 0.5}, {2., 2.1, 2.2, 2.3}, {1., 1.2, 1.3, 1.}){}

  OU::OU(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"mean", "theta", "phi", "noise_var"}){
      if (params.find(key) == params.end()){
        allParamsPresent=false;
        throw ConfigError("generator parameters don't have all required constructor arguments");
      }
    }
    if (allParamsPresent){
      vector<double> mean= std::any_cast<vector<double>>(params["mean"]);
      vector<double> theta= std::any_cast<vector<double>>(params["theta"]);
      vector<double> phi = std::any_cast<vector<double>>(params["phi"]);
      vector<double> noise_var = std::any_cast<vector<double>>(params["noise_var"]);
      initParams(mean, theta, phi, noise_var);
    }
    else{
      vector<double> mean{2., 4.3, 3., 0.5};
      vector<double> theta{1., 0.3, 2., 0.5};
      vector<double> phi{2., 2.1, 2.2, 2.3};
      vector<double> noise_var{1., 1.2, 1.3, 1.};
      initParams(mean, theta, phi, noise_var);
    }
  }

  OU::OU(pybind11::dict py_config): OU::OU(makeConfigFromPyDict(py_config)){}

  const PriceVector& OU::getData() {
    for (int i=0; i < nAssets_; i++){
      double& x = currentData_(i);
      x += theta[i]*(mean[i]-x)*dT + phi[i]*noiseDistribution[i](generator);
    }
    timestamp_ += 1;
    return currentData_;
  }

  void SimpleTrend::initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                               std::vector<int> maxPeriod, std::vector<double> noise,
                               std::vector<double> start, std::vector<double> dYMin,
                               std::vector<double> dYMax) {
    if ((trendProb.size() == minPeriod.size()) && (minPeriod.size() == maxPeriod.size())
        && (maxPeriod.size() == noise.size()) && (noise.size() == start.size())
        && (start.size() == dYMin.size()) && (dYMin.size() == dYMax.size())){
      this->trendProb=trendProb;
      this->minPeriod=minPeriod;
      this->maxPeriod=maxPeriod;
      this->noise=noise;
      this->dYMin=dYMin;
      this->dYMax=dYMax;
      currentData_.resize(trendProb.size());
      for (int i=0; i<trendProb.size(); i++){
        noiseDist.push_back(std::normal_distribution<double>(0., noise[i]));
        trendLenDist.push_back(std::uniform_int_distribution<int>
                                 (minPeriod[i], maxPeriod[i]));
        dYDist.push_back(std::uniform_real_distribution<double>
                                 (dYMin[i], dYMax[i]));
        trending.push_back(false);
        std::string assetName = "SimpleTrend_" + std::to_string(i);
        this->assets.push_back(Asset(assetName));
        currentData_ << start[i]; // default starting val
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
        dY.push_back(0.);
      }
      this->nAssets_ = assets.size();
    }
    else{
      throw std::length_error("parameters passed to DataSource of type SimpleTrend"
                              " need to be vectors of same length");
    }
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  SimpleTrend::SimpleTrend(std::vector<double> trendProb, std::vector<int> minPeriod,
                           std::vector<int> maxPeriod, std::vector<double> noise,
                           std::vector<double> start, std::vector<double> dYMin,
                           std::vector<double> dYMax) {
    initParams(trendProb, minPeriod, maxPeriod, noise, start, dYMin, dYMax);
  }

  SimpleTrend::SimpleTrend(): SimpleTrend({0.001, 0.001}, {100, 500}, {200, 1500},
                                          {1. ,0.1}, {10., 15.},
                                          {0.001, 0.01}, {0.003, 0.03}){}

  SimpleTrend::SimpleTrend(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"trendProb", "minPeriod", "maxPeriod", "noise", "start",
                    "dYMin", "dYMax"}){
      if (params.find(key) == params.end()){
        allParamsPresent=false;
        throw ConfigError("data_Source_config doesn't have all required constructor arguments");
      }
    }
    vector<double> trendProb = std::any_cast<vector<double>>(params["trendProb"]);
    vector<int> minPeriod = std::any_cast<vector<int>>(params["minPeriod"]);
    vector<int> maxPeriod = std::any_cast<vector<int>>(params["maxPeriod"]);
    vector<double> noise = std::any_cast<vector<double>>(params["noise"]);
    vector<double> dYMin = std::any_cast<vector<double>>(params["dYMin"]);
    vector<double> dYMax = std::any_cast<vector<double>>(params["dYMax"]);
    vector<double> start = std::any_cast<vector<double>>(params["start"]);
    initParams(trendProb, minPeriod, maxPeriod, noise, start, dYMin, dYMax);
  }

  SimpleTrend::SimpleTrend(pybind11::dict py_config):
    SimpleTrend::SimpleTrend(makeConfigFromPyDict(py_config)){}

  const PriceVector& SimpleTrend::getData() {
    for (int i=0; i < nAssets_; i++){
      double& y = currentData_[i];
      if(trending[i]){
        y += y * dY[i] * currentDirection[i];
        currentTrendLen[i] -= 1;
        if (currentTrendLen[i] == 0){
          trending[i] = false;
        }
      }
      else{
        double rand = uniformDist(generator);
        if (rand < trendProb[i]){
          trending[i] = true;
          currentDirection[i] = (uniformDist(generator) < 0.5)? -1: 1;
          currentTrendLen[i] = trendLenDist[i](generator);
          dY[i] = dYDist[i](generator);
        }
      }
      if (y <= .1){
        currentDirection[i] = 1;
      }
      y += y*noiseDist[i](generator);
      y = std::max(0.01, y);
    }
    timestamp_ += 1;
    return currentData_;
  }



  void TrendOU::initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                           std::vector<int> maxPeriod, std::vector<double> noise,
                           std::vector<double> dYMin, std::vector<double> dYMax,
                           std::vector<double> start, std::vector<double> theta,
                           std::vector<double> phi, std::vector<double> noise_var,
                           std::vector<double> emaAlpha){
    if ((trendProb.size() == minPeriod.size()) && (minPeriod.size() == maxPeriod.size())
        && (maxPeriod.size() == noise.size()) && (noise.size() == start.size())
        && (start.size() == dYMin.size()) && (dYMin.size() == dYMax.size())
        && (dYMax.size() == theta.size()) && (theta.size() == phi.size())
        && (phi.size() == noise_var.size())){
      this->trendProb=trendProb;
      this->minPeriod=minPeriod;
      this->maxPeriod=maxPeriod;
      this->noise=noise;
      this->dYMin=dYMin;
      this->dYMax=dYMax;
      this->theta=theta;
      this->phi=phi;
      this->noise_var=noise_var;
      this->emaAlpha=emaAlpha;
      this->ema=start;;
      currentData_.resize(trendProb.size());
      for (int i=0; i<trendProb.size(); i++){
        trendNoise.push_back(std::normal_distribution<double>(0., noise[i]));
        ouNoise.push_back(std::normal_distribution<double>(0., noise_var[i]));
        trendLenDist.push_back(std::uniform_int_distribution<int>
                                 (minPeriod[i], maxPeriod[i]));
        dYDist.push_back(std::uniform_real_distribution<double>
                                 (dYMin[i], dYMax[i]));
        trending.push_back(false);
        std::string assetName = "TrendOU_" + std::to_string(i);
        this->assets.push_back(Asset(assetName));
        currentData_ << start[i]; // default starting val
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
        dY.push_back(0.);
      }
      this->nAssets_ = assets.size();
    }
    else{
      throw std::length_error("parameters passed to DataSource of type TrendOU"
                              " need to be vectors of same length");
    }
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  TrendOU::TrendOU(std::vector<double> trendProb, std::vector<int> minPeriod,
                   std::vector<int> maxPeriod, std::vector<double> noise,
                   std::vector<double> dYMin, std::vector<double> dYMax,
                   std::vector<double> start, std::vector<double> theta,
                   std::vector<double> phi, std::vector<double> noise_var,
                   std::vector<double> emaAlpha) {
    initParams(trendProb, minPeriod, maxPeriod, noise, dYMin, dYMax, start,
               theta, phi, noise_var, emaAlpha);
  }

  TrendOU::TrendOU(): TrendOU({0.001, 0.001}, {100, 500}, {200, 1500},
                              {1. ,0.1}, {0.001, 0.01},
                              {0.003, 0.03}, {10., 15.},
                              {1., 0.5}, {2., 2.1},
                              {1., 1.2}, {0.1, 0.2}){}

  TrendOU::TrendOU(Config config){
    bool allParamsPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allParamsPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"trendProb", "minPeriod", "maxPeriod", "noise", "dYMin",
                    "dYMax", "start", "theta", "phi", "noise_var", "emaAlpha"}){
      if (params.find(key) == params.end()){
        allParamsPresent=false;
        throw ConfigError((string)"data_Source_config doesn't have all required"
                          "constructor arguments, missing: " + key );
      }
    }
    vector<double> trendProb = std::any_cast<vector<double>>(params["trendProb"]);
    vector<int> minPeriod = std::any_cast<vector<int>>(params["minPeriod"]);
    vector<int> maxPeriod = std::any_cast<vector<int>>(params["maxPeriod"]);
    vector<double> noise = std::any_cast<vector<double>>(params["noise"]);
    vector<double> dYMin = std::any_cast<vector<double>>(params["dYMin"]);
    vector<double> dYMax = std::any_cast<vector<double>>(params["dYMax"]);
    vector<double> start = std::any_cast<vector<double>>(params["start"]);
    vector<double> theta = std::any_cast<vector<double>>(params["theta"]);
    vector<double> phi = std::any_cast<vector<double>>(params["phi"]);
    vector<double> noise_var = std::any_cast<vector<double>>(params["noise_var"]);
    vector<double> emaAlpha = std::any_cast<vector<double>>(params["emaAlpha"]);
    initParams(trendProb, minPeriod, maxPeriod, noise, dYMin, dYMax, start,
               theta, phi, noise_var, emaAlpha);
  }

  TrendOU::TrendOU(pybind11::dict py_config):
    TrendOU::TrendOU(makeConfigFromPyDict(py_config)){}

  const PriceVector& TrendOU::getData(){
    for (int i=0; i < nAssets_; i++){
      double& y = currentData_[i];
      // OU PROCESS
      y += theta[i] * (ema[i]-y) * dT + y * phi[i] * ouNoise[i](generator);
      // TRENDING PROCESS
      if(trending[i]){
        double& x = currentData_(i);
        y += y * dY[i] * currentDirection[i];
        currentTrendLen[i] -= 1;
        if (currentTrendLen[i] == 0){
          trending[i] = false;
        }
        ema[i] += emaAlpha[i] * y + (1-emaAlpha[i]) * ema[i];
      }
      else{
        double rand = uniformDist(generator);
        if (rand < trendProb[i]){
          trending[i] = true;
          currentDirection[i] = (uniformDist(generator) < 0.5)? -1: 1;
          currentTrendLen[i] = trendLenDist[i](generator);
          dY[i] = dYDist[i](generator);
        }
      }
      if (y <= .1){
        currentDirection[i] = 1;
      }
      y += y*trendNoise[i](generator);
      y = std::max(0.01, y);
    }
    timestamp_ += 1;
    return currentData_;
  }




}// namespace madigan
