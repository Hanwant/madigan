#include <stdexcept>
#include <iostream>

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
    else{
      std::stringstream ss;
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
    else{
      std::stringstream ss;
      ss << dataSourceType;
      ss << " as dataSource is not implemented";
      throw NotImplemented(ss.str());
    }
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
    this->assets.push_back(Asset("multi_sine"));
    this->nAssets_ = assets.size();
    currentData_.resize(1);
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

}// namespace madigan
