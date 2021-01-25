#include <stdexcept>
#include <iostream>
#include <numeric>

#include "DataSource.h"

namespace madigan{

  template<>
  std::unique_ptr<DataSource<PriceVector>> makeDataSource(string dataSourceType){
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
    else if (dataSourceType == "SineDynamic"){
      return make_unique<SineDynamic>();
    }
    else if (dataSourceType == "SineDynamicTrend"){
      return make_unique<SineDynamicTrend>();
    }
    else if (dataSourceType == "Gaussian"){
      return make_unique<Gaussian>();
    }
    else if (dataSourceType == "OU"){
      return make_unique<OU>();
    }
    else if (dataSourceType == "OUPair"){
      return make_unique<OUPair>();
    }
    else if (dataSourceType == "SimpleTrend"){
      return make_unique<SimpleTrend>();
    }
    else if (dataSourceType == "TrendOU"){
      return make_unique<TrendOU>();
    }
    else if (dataSourceType == "TrendyOU"){
      return make_unique<TrendyOU>();
    }
    else{
      std::stringstream ss;
      ss << "Default Constructor for ";
      ss << dataSourceType;
      ss << " as dataSource is not implemented";
      throw NotImplemented(ss.str());
    }
  }

  template<>
  std::unique_ptr<DataSource<PriceVector>> makeDataSource(string dataSourceType,
                                                          Config config){
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
    else if (dataSourceType == "SineDynamic"){
      return make_unique<SineDynamic>(config);
    }
    else if (dataSourceType == "SineDynamicTrend"){
      return make_unique<SineDynamicTrend>(config);
    }
    else if (dataSourceType == "Gaussian"){
      return make_unique<Gaussian>(config);
    }
    else if (dataSourceType == "OU"){
      return make_unique<OU>(config);
    }
    else if (dataSourceType == "OUPair"){
      return make_unique<OUPair>(config);
    }
    else if (dataSourceType == "SimpleTrend"){
      return make_unique<SimpleTrend>(config);
    }
    else if (dataSourceType == "TrendOU"){
      return make_unique<TrendOU>(config);
    }
    else if (dataSourceType == "TrendyOU"){
      return make_unique<TrendyOU>(config);
    }
    else if (dataSourceType == "Composite"){
      return make_unique<Composite>(config);
    }
    else if (dataSourceType == "HDFSourceSingle"){
      return make_unique<HDFSourceSingle>(config);
    }
    else{
      std::stringstream ss;
      ss << "Constructor from config for";
      ss << dataSourceType;
      ss << " as dataSource is not implemented";
      throw NotImplemented(ss.str());
    }
  }

  vector<string> checkKeysPresent(Config config, vector<string> keys){
    // Returns missing keys (if any)
    vector<string> missing;
    for (auto key: keys){
      if (config.find(key) == config.end()){
        missing.push_back(key);
      }
    }
    return missing;
  }

  template<class T>  // T could be Eigen::Vector or vector<numeric>
  T loadVectorFromHDF(string fname, string groupKey, string vectorKey,
                      std::size_t start=0, std::size_t  size=-1){
    H5Easy::File file(fname, HighFive::File::ReadOnly);
    string datasetPath = "/"+groupKey+"/"+vectorKey;
    size_t len = H5Easy::getSize(file, datasetPath);
    if (size == -1) size = len;
    else{
      len = size;
    }
    T out(len);
    HighFive::DataSet dataset = file.getDataSet(datasetPath);
    dataset.select({start}, {size}).read(out.data());
    return out;
  }

  template<class T>  // T expected to be an EigenMatrix
  T loadMatrixFromHDF(string filename, string groupKey, string matrixKey,
                      std::size_t start=0, std::size_t  size=-1){
    H5Easy::File file(filename, HighFive::File::ReadOnly);
    string datasetPath = "/"+groupKey+"/"+matrixKey;
    HighFive::DataSet dataset = file.getDataSet(datasetPath);
    vector<size_t> shape = dataset.getSpace().getDimensions();
    if (size == -1) size = shape[0];
    if (size > shape[0]) size = shape[0];
    T out(size, shape[1]);
    dataset.select({start, 0}, {size, shape[1]}).read(out.data());
    return out;
  }

  template<class T>
  std::pair<T, T> getBounds(string filepath, string groupKey, string dsetKey){
    H5Easy::File file(filepath, HighFive::File::ReadOnly);
    string datasetPath = "/"+groupKey+"/"+dsetKey;
    // size_t len = H5Easy::getSize(file, datasetPath);
    HighFive::DataSet dataset = file.getDataSet(datasetPath);
    vector<size_t> shape = dataset.getSpace().getDimensions();
    std::pair<T, T> out;
    dataset.select({0}, {1}).read(&out.first);
    dataset.select({shape[0]-1}, {1}).read(&out.second);
    vector<T> full(shape[0]);
    dataset.read(full.data());
    return out;
  }

  template<class T>
  size_t binarySearchSortedHDFArray(string filepath, string key, T val){
    H5Easy::File file (filepath, HighFive::File::ReadOnly);
    HighFive::DataSet dataset = file.getDataSet(key);
    vector<size_t> shape = dataset.getSpace().getDimensions();
    int maxTries = log2(shape[0]) + 2;
    size_t l=0, r=shape[0], m;
    T buffer;
    while (l <= r && maxTries -- > 0){
      m = l + (r-l) / 2;
      dataset.select({m}, {1}).read(&buffer);
      if (buffer == val){
        return m;
      }
      if (buffer < val){
        l = m + 1;
      }
      else{
        r = m - 1;
      }
    }
    return m;
  }


  /////////////////////////////////////////////////////////////////////////////////////////
  // HDF SOURCE  //////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  void HDFSourceSingle::init(){
    checkKeys();
    loadAssets();
    loadDimsInfo();
    loadData();
  }

  void HDFSourceSingle::reset(){
    // Carry on
    // currentIdx_ = 0;
    // currentCacheIdx_ = 0;
    // loadData();
  }

  HDFSourceSingle::HDFSourceSingle(string filepath, string groupKey, string priceKey,
                                   string featureKey, string timestampKey,
                                   std::size_t cacheSize):
    filepath(filepath),
    groupKey(groupKey),
    priceKey(priceKey),
    featureKey(featureKey),
    timestampKey(timestampKey),
    cacheSize(cacheSize)
  {
    init();
  }

  HDFSourceSingle::HDFSourceSingle(string filepath, string groupKey, string priceKey,
                                   string featureKey, string timestampKey,
                                   std::size_t cacheSize, std::size_t startTime,
                                   std::size_t endTime):
    filepath(filepath),
    groupKey(groupKey),
    priceKey(priceKey),
    featureKey(featureKey),
    timestampKey(timestampKey),
    cacheSize(cacheSize),
    startTime(startTime),
    endTime(endTime)
  {
    init();
  }

  HDFSourceSingle::HDFSourceSingle(Config config){
    if (checkKeysPresent(config, {"data_source_config"}).size() !=0){
      throw ConfigError("Key: data_source_config not in Config");
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    vector<string> missing =
      checkKeysPresent(params, {"filepath", "group_key", "feature_key",
                                "timestamp_key", "price_key", "cache_size"});
    if (missing.size() == 0){
      filepath = std::any_cast<string>(params["filepath"]);
      groupKey = std::any_cast<string>(params["group_key"]);
      timestampKey = std::any_cast<string>(params["timestamp_key"]);
      priceKey= std::any_cast<string>(params["price_key"]);
      featureKey= std::any_cast<string>(params["feature_key"]);
      cacheSize = (std::size_t) std::any_cast<long long int>(params["cache_size"]);
      if (checkKeysPresent(params, {"start_time", "end_time"}).size()==0){
        startTime = (std::size_t) std::any_cast<long long int>(params["start_time"]);
        endTime = (std::size_t) std::any_cast<long long int>(params["end_time"]);
      }
      // Main Init
      init();
    }
    else{
      stringstream msg;
      msg << "Missing keys in call to HDFSourceSingle: \n";
      for (auto& missing_key: missing){
        msg << missing_key << ", ";
      }
      msg << "\n";
      throw ConfigError(msg.str());
    }
  }

  void checkHDFObjForKey(const vector<string>& names, const string& target,
                         const string& filepath, const string& searchLoc){
    auto found = std::find(names.begin(), names.end(), target);
    if (found == names.end()){
      stringstream msg;
      msg  << "key: " << target << "not found in " << searchLoc << " in hdfile: ";
      msg << filepath << "\n";
      throw ConfigError(msg.str());
    }
  }


  void HDFSourceSingle::checkKeys(){
    H5Easy::File file (filepath, HighFive::File::ReadOnly);
    HighFive::Group group = file.getGroup(groupKey);
    vector<string> objects = group.listAttributeNames();
    checkHDFObjForKey(objects, "assets", filepath, groupKey);
    objects = group.listObjectNames();
    checkHDFObjForKey(objects, priceKey, filepath, groupKey);
    checkHDFObjForKey(objects, featureKey, filepath, groupKey);
    checkHDFObjForKey(objects, timestampKey, filepath, groupKey);
  }

  void HDFSourceSingle::loadAssets(){
    H5Easy::File file (filepath, HighFive::File::ReadOnly);
    HighFive::Group group = file.getGroup(groupKey);
    HighFive::Attribute attr = group.getAttribute("assets");
    vector<std::size_t> dims = attr.getSpace().getDimensions();
    vector<string> assetCodes(dims[0]);
    attr.read(assetCodes);
    assets_ = Assets(assetCodes);
  }


  void HDFSourceSingle::loadDimsInfo(){
    vector<std::size_t> dims;
    H5Easy::File file (filepath, HighFive::File::ReadOnly);
    dims = H5Easy::getShape(file, groupKey+'/'+featureKey);
    // HighFive::Group group = file.getGroup(groupKey);
    // vector<std::size_t> dims = group.getDataSet(priceKey).getSpace().getDimensions();

    getTimeBounds();
    findBounds();

    fullDataSetLen_ = boundsIdx_.second - boundsIdx_.first;
    currentIdx_ = boundsIdx_.first;
    nFeats_ = dims[1];
    cacheSize = min(cacheSize, fullDataSetLen_);
    // data_.resize(min(cacheSize, fullDataSetLen_), nfeats_);
    currentData_.resize(nFeats_);
    currentPrices_.resize(1);
  }

  void HDFSourceSingle::getTimeBounds(){
    std::pair<size_t, size_t> bounds = getBounds<size_t>(filepath, groupKey, timestampKey);
    if (startTime == 0 && endTime == 0){  // get time bounds from data
      // std::pair<size_t, size_t> bounds = getTimeBounds();
      startTime = bounds.first;
      endTime = bounds.second;
    }
    else{  // validate given timebounds
      if (!(startTime >= bounds.first && endTime <= bounds.second)){
        throw std::out_of_range("Given start and endTimes not "
                                "within bounds found in timestamp data");
      }
    }
  }

  void HDFSourceSingle::findBounds(){
    // Does Approximate Binary Seach. For startTime and endTime in timestamps
    // By First doing a normal binary Search and then adjusting if not exact match
    size_t startIdx = binarySearchSortedHDFArray(filepath, groupKey+'/'+timestampKey,
                                                 startTime);
    size_t endIdx = binarySearchSortedHDFArray(filepath, groupKey+'/'+timestampKey,
                                               endTime);
    if (endIdx - startIdx < 2){
      stringstream msg;
      msg << "dset size only " << endIdx - startIdx << " !\n";
      msg << "with startIdx, startTime: " << startIdx << ", " << startTime << "\n";
      msg << "and endIdx, endTime: " << endIdx << ", " << endTime << "\n";
      throw std::length_error(msg.str());
    }

    HighFive::File file(filepath, HighFive::File::ReadOnly);
    HighFive::DataSet dataset = file.getDataSet(groupKey+'/'+timestampKey);
    vector<size_t> dims = dataset.getSpace().getDimensions();
    size_t buff;

    dataset.select({startIdx}, {1}).read(&buff);
    if (buff == startTime || startIdx == 0){
      boundsIdx_.first = startIdx;
    }else boundsIdx_.first = (buff < startTime)? (startIdx+1): startIdx;
    dataset.select({endIdx}, {1}).read(&buff);
    if (buff == endTime || endIdx == dims[0] - 1){
      boundsIdx_.second = endIdx;
    }else boundsIdx_.second = (buff > endTime)? (endIdx-1): endIdx; // end not included
  }

  void HDFSourceSingle::loadData(){
    if (currentIdx_ >= boundsIdx_.second-1 ){
      currentIdx_ = boundsIdx_.first;
    }
    currentCacheSize_ = min(cacheSize, boundsIdx_.second - currentIdx_);
    prices_ = loadVectorFromHDF<PriceVector>(filepath, groupKey, priceKey,
                                             currentIdx_, currentCacheSize_);
    data_ = loadMatrixFromHDF<PriceMatrix>(filepath, groupKey, featureKey,
                                               currentIdx_, currentCacheSize_);
    timestamps_ = loadVectorFromHDF<TimeVector>(filepath, groupKey, timestampKey,
                                                currentIdx_, currentCacheSize_);
  }

  const PriceVector& HDFSourceSingle::getData(){
    iterCache();
    currentPrices_(0) = prices_(currentCacheIdx_);
    currentData_ = data_.row(currentCacheIdx_);
    timestamp_ = timestamps_(currentCacheIdx_);
    currentIdx_++;
    currentCacheIdx_++;
    return currentData_;
  }

  void HDFSourceSingle::iterCache(){
    // replenishes cache if required
    // Cycles data back to beginning if reached end
    // currentIdx_++;
    // currentCacheIdx_++;
    if (currentCacheIdx_ == currentCacheSize_ || currentIdx_ == boundsIdx_.second){
      if (currentCacheSize_ >= fullDataSetLen_){
        currentIdx_ = boundsIdx_.first;
        currentCacheIdx_ = 0;
      }else{
        loadData();
        currentCacheIdx_ = 0;
      }
    }
    // For negative indices:
    // currentIdx_ = (((currentIdx_ + 1) % prices_.size()) + prices_.size()) % prices_.size();

  }

  // COMPOSITE SOURCE  //////////////////////////////////////////////////////////////////////////
  Composite::Composite(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto singleSource: params){
      string dataSourceType = std::any_cast<string>(singleSource.first);
      Config config = std::any_cast<Config>(singleSource.second);
      std::unique_ptr<DataSource> source = makeDataSource<PriceVector>(dataSourceType, config);
      dataSources_.emplace_back(std::move(source));
    }
    nFeats_ = 0;
    for(const auto& source: dataSources_){
      nFeats_ += source->nFeats();
      for (const auto& asset: source->assets()){
        if (std::find(assets_.begin(), assets_.end(), asset) != assets_.end()){
          assets_.emplace_back(Asset(asset.code + "_1"));
        }
        else assets_.emplace_back(asset);
      }
    }
    currentPrices_.resize(nAssets());
    currentData_.resize(nFeats_);
    // currentPrices_.resize(nAssets());
  }

  const PriceVector& Composite::getData(){
    int featIdx{0};
    int priceIdx{0};
    for (int i=0; i<dataSources_.size(); i++){
      DataSource* source = dataSources_[i].get();
      currentData_.segment(featIdx, source->nFeats()) = source->getData();
      currentPrices_.segment(priceIdx, source->nAssets()) = source->currentPrices();
      featIdx += source->nFeats();
      priceIdx += source->nAssets();
    }
    timestamp_ += 1;
    return currentData_;
  }

  // SYNTH SOURCE  //////////////////////////////////////////////////////////////////////////

  void Synth::initParams(vector<double> _freq, vector<double> _mu,
                         vector<double> _amp, vector<double> _phase,
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
      this->assets_.push_back(Asset(assetName));
    }
    currentData_.resize(nAssets());;
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

  Synth::Synth(vector<double> _freq, vector<double> _mu,
               vector<double> _amp, vector<double> _phase,
               double _dX, double noise){
    if ((_freq.size() == _mu.size()) && (_mu.size() == _amp.size()) &&
        (_amp.size() == _phase.size())){
      initParams(_freq, _mu, _amp, _phase, _dX, noise);
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type Synth"
                              " need to be vectors of same length");
    }
  }

  Synth::Synth(Config config){
    if (checkKeysPresent(config, {"data_source_config"}).size() !=0){
      throw ConfigError("Key: data_source_config not in Config");
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    vector<string> missing =
      checkKeysPresent(params, {"freq", "mu", "amp", "phase", "dX"});
    if (missing.size() == 0){
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
      stringstream msg;
      msg << "Missing keys in call to Synth: \n";
      for (auto& missing_key: missing){
        msg << missing_key << ", ";
      }
      msg << "\n";
      throw ConfigError(msg.str());
    }
  }

  Synth::Synth(pybind11::dict py_config): Synth::Synth(makeConfigFromPyDict(py_config)){
    // Config config = makeConfigFromPyDict(py_config);

  }

  const PriceVector& Synth::getData() {
    for (int i=0; i < nAssets(); i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }

  const pybind11::array_t<double> Synth::getData_np(){
    for (int i=0; i < nAssets(); i++){
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
    for (int i=0; i < nAssets(); i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        amp[i] * std::modf(x[i]*freq[i], &intPart);
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }

  const PriceVector& Triangle::getData() {
    double intPart;
    for (int i=0; i < nAssets(); i++){
      currentData_[i] = noiseDistribution(generator) + mu[i] +
        4*amp[i]/PI2 * std::asin(std::sin(PI2*x[i]/freq[i]));
      x[i] += dX;
    }
    timestamp_ += 1;
    return currentData_;
  }


  // SINE ADDER  ///////////////////////////////////////////////////////////////////////////////////
  SineAdder::SineAdder(){
    double _dX{0.01};
    vector<double> freq{1., 0.3, 2., 0.5};
    vector<double> mu{2., 2.1, 2.2, 2.3};
    vector<double> amp{1., 1.2, 1.3, 1.};
    vector<double> phase{0., 1., 2., 1.};
    initParams(freq, mu, amp, phase, dX, 0.);
  }

  SineAdder::SineAdder(vector<double> _freq, vector<double> _mu,
                       vector<double> _amp, vector<double> _phase,
                       double _dX, double noise){
    if ((_freq.size() == _mu.size()) && (_mu.size() == _amp.size()) &&
        (_amp.size() == _phase.size())){
      initParams(_freq, _mu, _amp, _phase, _dX, noise);
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type SineAdder"
                              " need to be vectors of same length");
    }
  }

  SineAdder::SineAdder(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"freq", "mu", "amp", "phase", "dX"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config doesn't don't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
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
  void SineAdder::initParams(vector<double> _freq, vector<double> _mu,
                             vector<double> _amp, vector<double> _phase,
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
    this->assets_ = vector<Asset>(1, Asset("multi_sine"));
    // this->nAssets() = this->assets.size();
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

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // SINE DYNAMIC  ///////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  SineDynamic::SineDynamic(){
    vector<std::array<double, 3>> freqRange{{.1, 1., .01}, {0.3, 3.0, .01},
                                                    {5., 15., .1}, {10., 50., .1}};
    vector<std::array<double, 3>> muRange{{1., 5., .02}, {.3, 3., .05},
                                                  {.2, 5., .02}, {.5, 5., .02}};
    vector<std::array<double, 3>> ampRange{{1., 5., .01}, {.3, 3., .02},
                                                   {.2, 2., .04}, {.5, 5., .05}};
    double _dX{0.01};
    initParams(freqRange, muRange, ampRange, dX, 1.);
  }

  SineDynamic::SineDynamic(vector<std::array<double, 3>> _freqRange,
                           vector<std::array<double, 3>> _muRange,
                           vector<std::array<double, 3>> _ampRange,
                           double _dX, double noise){
    initParams(_freqRange, _muRange, _ampRange, _dX, noise);
  }

  SineDynamic::SineDynamic(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"freqRange", "muRange", "ampRange", "dX"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config doesn't don't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
      vector<std::array<double, 3>> freqRange =
        std::any_cast<vector<std::array<double, 3>>>(params["freqRange"]);
      vector<std::array<double, 3>> muRange =
        std::any_cast<vector<std::array<double, 3>>>(params["muRange"]);
      vector<std::array<double, 3>> ampRange =
        std::any_cast<vector<std::array<double, 3>>>(params["ampRange"]);
      double dX = std::any_cast<double>(params["dX"]);
      if (params.find("noise") != params.end()){
        noise = std::any_cast<double>(params["noise"]);
        initParams(freqRange, muRange, ampRange, dX, noise);
     }
      else {
        initParams(freqRange, muRange, ampRange, dX, 0.);
      }

    }
    else{
      vector<std::array<double, 3>> freqRange{{.1, 1., .01}, {0.3, 3.0, .01},
                                                      {5., 15., .1}, {10., 50., .1}};
      vector<std::array<double, 3>> muRange{{1., 5., .02}, {.3, 3., .05},
                                                    {.2, 5., .02}, {.5, 5., .02}};
      vector<std::array<double, 3>> ampRange{{1., 5., .01}, {.3, 3., .02},
                                                     {.2, 2., .04}, {.5, 5., .05}};
      double _dX{0.01};
      initParams(freqRange, muRange, ampRange, _dX, 0.);
    }
  }

  SineDynamic::SineDynamic(pybind11::dict py_config): SineDynamic::SineDynamic(makeConfigFromPyDict(py_config)){
    // Config config = makeConfigFromPyDict(py_config);

  }
  void SineDynamic::initParams(vector<std::array<double, 3>> _freqRange,
                               vector<std::array<double, 3>> _muRange,
                               vector<std::array<double, 3>> _ampRange,
                               double _dX, double _noise)

  {
    if ((_freqRange.size() == _muRange.size()) && (_muRange.size() == _ampRange.size())){
      nComponents = _freqRange.size();
      freqRange=_freqRange;
      muRange=_muRange;
      ampRange=_ampRange;
      dX=_dX;
      noise=_noise;
      noiseDistribution = std::normal_distribution<double>(0., _noise);
      updateParameterDist = std::uniform_real_distribution<double>(0., 1.);
      assets_ = vector<Asset>(1, Asset("sine_dynamic"));
      currentData_.resize(1);
      freq = vector<double>(nComponents, 0);
      mu= vector<double>(nComponents, 0);
      amp= vector<double>(nComponents, 0);
      generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
      sampleRate = (int)(1./dX);
      oscillators.resize(nComponents); // so resizing doesn't invalidate pointers
      assets_ = Assets({"sine_dynamic"});
      // x.emplace_back(0.);
      for (int i=0; i<nComponents; i++){
        // Init Starting values by uniformly sampling from low-high range
        if (freqRange[i][1] * 2 > sampleRate){
          std::stringstream msg;
          msg << "Sampling Rate as determined by 1 / dX must be at least the ";
          msg << "nyquist sampling rate relative to the largest frequency. ";
          msg << "freqRange entry #" << i << " with range of " << freqRange[i][0];
          msg << " - " << freqRange[i][1] << " doesn't satisfy this constraint.\n";
          throw std::logic_error(msg.str());
        }
        freqDist.push_back(std::uniform_real_distribution<double>(freqRange[i][0], freqRange[i][1]));
        muDist.push_back(std::uniform_real_distribution<double>(muRange[i][0], muRange[i][1]));
        ampDist.push_back(std::uniform_real_distribution<double>(ampRange[i][0], ampRange[i][1]));
        freq[i] = freqDist[i](generator);
        mu[i] = muDist[i](generator);
        amp[i] = ampDist[i](generator);
        int maxNumTables = log2(ceil(freqRange[i][1] / freqRange[i][0])+1);
        oscillators.emplace_back(WaveTableOsc<double>(maxNumTables));
        setSineOsc(oscillators[i], sampleRate, 2, freqRange[i][0]);
      }
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type SineDynamic"
                              " need to be vectors of same length");
    }
  }

  void SineDynamic::reset(){
    for (int i=0; i<nComponents; i++){
      freq[i] = freqDist[i](generator);
      mu[i] = muDist[i](generator);
      amp[i] = ampDist[i](generator);
    }
  }

  void SineDynamic::updateParams(){
    for (int i=0; i<nComponents; i++){
      mu[i] = max(muRange[i][0],
                  min(muRange[i][1],
                      mu[i]+(boolDist.randBool()? muRange[i][2]: -muRange[i][2])));
      amp[i] = max(ampRange[i][0],
                   min(ampRange[i][1],
                       amp[i]+(boolDist.randBool()? ampRange[i][2]: -ampRange[i][2])));
      freq[i] = max(freqRange[i][0],
                    min(freqRange[i][1],
                        freq[i]+(boolDist.randBool()? freqRange[i][2]: -freqRange[i][2])));
      oscillators[i].setFreq(freq[i]/sampleRate);
      // if (updateParameterDist(generator) < 0.001 ){
      //   // mu[i] = max(muRange[i][0],
      //   //             min(muRange[i][1],
      //   //                 mu[i]+(boolDist.randBool()? muRange[i][2]: -muRange[i][2])));
      //   // amp[i] = max(ampRange[i][0],
      //   //              min(ampRange[i][1],
      //   //                  amp[i]+(boolDist.randBool()? ampRange[i][2]: -ampRange[i][2])));
      //   // freq[i] = freqDist[i](generator);
      //   mu[i] = muDist[i](generator);
      //   amp[i] = ampDist[i](generator);
      //   stepsSinceUpdate = 0;
      // } else stepsSinceUpdate++;
    }
  }

  const PriceVector& SineDynamic::getData() {
    double sum{0.};
    updateParams();
    for (int i=0; i < nComponents; i++){
      // sum += noiseDistribution(generator) + mu[i] +
      //   amp[i] * std::sin(PI2*x[i]*freq[i]);
      sum +=  mu[i] + amp[i] * oscillators[i].process();
      // x[i] += dX;
    }
    currentData_[0] = sum + noiseDistribution(generator);
    timestamp_ += 1;
    return currentData_;
  }

  double SineDynamic::getProcess(int i){
    return oscillators[i].process();
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // SINE DYNAMIC TREND   ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  SineDynamicTrend::SineDynamicTrend(){
    vector<std::array<double, 3>> freqRange{{.1, 1., .01}, {0.3, 3.0, .01},
                                            {5., 15., .1}, {10., 50., .1}};
    vector<std::array<double, 3>> muRange{{1., 5., .02}, {.3, 3., .05},
                                          {.2, 5., .02}, {.5, 5., .02}};
    vector<std::array<double, 3>> ampRange{{1., 5., .01}, {.3, 3., .02},
                                           {.2, 2., .04}, {.5, 5., .05}};
    vector<std::array<int, 2>> trendRange{{100, 500}, {100, 300}};
    vector<double> trendIncr{0.1, 0.2};
    vector<double> trendProb{.001, .01};
    double _dX{0.01};
    initParams(freqRange, muRange, ampRange, trendRange, trendIncr, trendProb, dX, 1.);
  }

  SineDynamicTrend::SineDynamicTrend(vector<std::array<double, 3>> _freqRange,
                                     vector<std::array<double, 3>> _muRange,
                                     vector<std::array<double, 3>> _ampRange,
                                     vector<std::array<int, 2>> _trendRange,
                                     vector<double> _trendIncr,
                                     vector<double> _trendProb,
                                     double _dX, double noise){
    initParams(_freqRange, _muRange, _ampRange, _trendRange, _trendIncr,
               _trendProb, _dX, noise);
  }

  SineDynamicTrend::SineDynamicTrend(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"freqRange", "muRange", "ampRange", "trendRange", "trendIncr",
                    "trendProb", "dX", "noise"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config doesn't don't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
      vector<std::array<double, 3>> freqRange =
        std::any_cast<vector<std::array<double, 3>>>(params["freqRange"]);
      vector<std::array<double, 3>> muRange =
        std::any_cast<vector<std::array<double, 3>>>(params["muRange"]);
      vector<std::array<double, 3>> ampRange =
        std::any_cast<vector<std::array<double, 3>>>(params["ampRange"]);
      vector<std::array<int, 2>> trendRange =
        std::any_cast<vector<std::array<int, 2>>>(params["trendRange"]);
      vector<double> trendIncr = std::any_cast<vector<double>>(params["trendIncr"]);
      vector<double> trendProb = std::any_cast<vector<double>>(params["trendProb"]);
      double dX = std::any_cast<double>(params["dX"]);
      noise = std::any_cast<double>(params["noise"]);
      initParams(freqRange, muRange, ampRange, trendRange, trendIncr, trendProb,
                 dX, noise);

    }
    else{
      vector<std::array<double, 3>> freqRange{{.1, 1., .01}, {0.3, 3.0, .01},
                                                      {5., 15., .1}, {10., 50., .1}};
      vector<std::array<double, 3>> muRange{{1., 5., .02}, {.3, 3., .05},
                                                    {.2, 5., .02}, {.5, 5., .02}};
      vector<std::array<double, 3>> ampRange{{1., 5., .01}, {.3, 3., .02},
                                                     {.2, 2., .04}, {.5, 5., .05}};
      vector<std::array<int, 2>> trendRange{{100, 500}, {100, 300}};
      vector<double> trendIncr{0.1, 0.2};
      vector<double> trendProb{.001, .01};
      double _dX{0.01};
      initParams(freqRange, muRange, ampRange, trendRange, trendIncr, trendProb, dX, 1.);
    }
  }

  SineDynamicTrend::SineDynamicTrend(pybind11::dict py_config): SineDynamicTrend::SineDynamicTrend(makeConfigFromPyDict(py_config)){
    // Config config = makeConfigFromPyDict(py_config);

  }
  void SineDynamicTrend::initParams(vector<std::array<double, 3>> _freqRange,
                                    vector<std::array<double, 3>> _muRange,
                                    vector<std::array<double, 3>> _ampRange,
                                    vector<std::array<int, 2>> _trendRange,
                                    vector<double> _trendIncr,
                                    vector<double>_trendProb,
                                    double _dX, double _noise)

  {
    if (!((_freqRange.size() == _muRange.size()) && (_muRange.size() == _ampRange.size()))){
      throw std::length_error("the following parameters passed to DataSource<PriceVector>"
                              "of type SineDynamicTrend need to be vectors of same length:"
                              "freqRange\n muRange\n ampRange\n");
    }
    if (!((_trendRange.size() == _trendIncr.size()) && (_trendIncr.size() == _trendProb.size()))){
      throw std::length_error("the following parameters passed to DataSource<PriceVector>"
                              "of type SineDynamicTrend need to be vectors of same length:"
                              "trendRange\n trendIncr\n trendProb\n");
    }
      nComponents = _freqRange.size();
      freqRange = _freqRange;
      muRange = _muRange;
      ampRange = _ampRange;
      trendRange = _trendRange;
      trendIncr = _trendIncr;
      trendProb = _trendProb;
      dX = _dX;
      noise = _noise;
      noiseDistribution = std::normal_distribution<double>(0., _noise);
      updateParameterDist = std::uniform_real_distribution<double>(0., 1.);
      assets_ = vector<Asset>(1, Asset("sine_dynamic_trend"));
      currentData_.resize(1);
      freq = vector<double>(nComponents, 0);
      mu = vector<double>(nComponents, 0);
      amp = vector<double>(nComponents, 0);
      generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
      sampleRate = (int)(1./dX);
      oscillators.resize(nComponents); // so resizing doesn't invalidate pointers
      // x.emplace_back(0.);
      for (int i=0; i<nComponents; i++){
        // Init Starting values by uniformly sampling from low-high range
        if (freqRange[i][1] * 2 > sampleRate){
          std::stringstream msg;
          msg << "Sampling Rate as determined by 1 / dX must be at least the ";
          msg << "nyquist sampling rate relative to the largest frequency. ";
          msg << "freqRange entry #" << i << " with range of " << freqRange[i][0];
          msg << " - " << freqRange[i][1] << " doesn't satisfy this constraint.\n";
          throw std::logic_error(msg.str());
        }
        freqDist.push_back(std::uniform_real_distribution<double>(freqRange[i][0], freqRange[i][1]));
        muDist.push_back(std::uniform_real_distribution<double>(muRange[i][0], muRange[i][1]));
        ampDist.push_back(std::uniform_real_distribution<double>(ampRange[i][0], ampRange[i][1]));
        freq[i] = freqDist[i](generator);
        mu[i] = muDist[i](generator);
        amp[i] = ampDist[i](generator);
        int maxNumTables = log2(ceil(freqRange[i][1] / freqRange[i][0])+1);
        oscillators.emplace_back(WaveTableOsc<double>(maxNumTables));
        setSineOsc(oscillators[i], sampleRate, 2, freqRange[i][0]);
      }
      trendProbDist = std::uniform_real_distribution<double>(0., 1.);
      trendComponent = 1.;
      for (int i=0; i<trendProb.size(); i++){
        trending.push_back(false);
        trendLenDist.push_back(std::uniform_int_distribution<int>(trendRange[i][0], trendRange[i][1]));
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
      }
  }

  void SineDynamicTrend::reset(){
    for (int i=0; i<nComponents; i++){
      freq[i] = freqDist[i](generator);
      mu[i] = muDist[i](generator);
      amp[i] = ampDist[i](generator);
    }
  }

  void SineDynamicTrend::updateParams(){
    for (int i=0; i<nComponents; i++){
      mu[i] = max(muRange[i][0],
                  min(muRange[i][1],
                      mu[i]+(boolDist.randBool()? muRange[i][2]: -muRange[i][2])));
      amp[i] = max(ampRange[i][0],
                   min(ampRange[i][1],
                       amp[i]+(boolDist.randBool()? ampRange[i][2]: -ampRange[i][2])));
      freq[i] = max(freqRange[i][0],
                    min(freqRange[i][1],
                        freq[i]+(boolDist.randBool()? freqRange[i][2]: -freqRange[i][2])));
      oscillators[i].setFreq(freq[i]/sampleRate);
    }
  }

  const PriceVector& SineDynamicTrend::getData() {
    double sum{0.};
    updateParams();
    double& currentVal = currentData_[0];
    for (int i=0; i < nComponents; i++){
      sum +=  trendComponent * (mu[i] + amp[i] * oscillators[i].process());
    }
    for (int i=0; i < trendProb.size(); i++){
      if (trending[i]){
        trendComponent += trendComponent * trendIncr[i] * currentDirection[i];
        if (--currentTrendLen[i] == 0){
          trending[i] = false;
        }
      }
      else{
        double rand = trendProbDist(generator);
        if (rand < trendProb[i]){
          trending[i] = true;
          currentDirection[i] = (boolDist.randBool())? -1: 1;
          currentTrendLen[i] = trendLenDist[i](generator);
        }
      }
      if (trendComponent <= .1){
        currentDirection[i] = 1;
      }
      trendComponent = std::max(0.01, trendComponent);
    }
    currentData_[0] = sum +  trendComponent + trendComponent*noiseDistribution(generator);
    timestamp_ += 1;
    return currentData_;
  }

  double SineDynamicTrend::getProcess(int i){
    return oscillators[i].process();
  }


  ///////////////////////////////////////////////////////////////////////////////////////////
  // Gaussian ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////
  void Gaussian::initParams(vector<double> mean, vector<double> var){
    if (mean.size() == var.size()){
      this->mean=mean;
      this->var=var;
      currentData_.resize(mean.size());
      for (int i=0; i < mean.size(); i++){
        std::string assetName = "Gaussian_" + std::to_string(i);
        this->assets_.push_back(Asset(assetName));
        noiseDistribution.push_back(std::normal_distribution<double>(mean[i], var[i]));
        currentData_(i) = mean[i];
      }
      generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type Gaussian"
                              " need to be vectors of same length");
    }
  }

  Gaussian::Gaussian(vector<double> mean, vector<double> var){
    initParams(mean, var);
  }
  Gaussian::Gaussian(): Gaussian({2., 5, 10., 15}, {1., 1., 2., 5.}){}

  Gaussian::Gaussian(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"mean", "var"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config doesn't don't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
      vector<double> mean = std::any_cast<vector<double>>(params["mean"]);
      vector<double> var = std::any_cast<vector<double>>(params["var"]);
      initParams(mean, var);
    }
    else{
      vector<double> mean{2., 5., 10., 15.};
      vector<double> var{1., 1., 2., 5.};
      initParams(mean, var);
    }
  }

  Gaussian::Gaussian(pybind11::dict py_config): Gaussian::Gaussian(makeConfigFromPyDict(py_config)){}

  const PriceVector& Gaussian::getData() {
    for (int i=0; i < nAssets(); i++){
      currentData_(i) = noiseDistribution[i](generator);
    }
    timestamp_ += 1;
    return currentData_;
  }


  // OU ///////////////////////////////////////////////////////////////////////////////
  void OU::initParams(vector<double> mean, vector<double> theta,
                      vector<double> phi) {
    if ((mean.size() == theta.size()) && (theta.size() == phi.size())){
      this->mean=mean;
      this->theta=theta;
      this->phi=phi;
      currentData_.resize(mean.size());
      for (int i=0; i < mean.size(); i++){
        std::string assetName = "OU_" + std::to_string(i);
        this->assets_.push_back(Asset(assetName));
        noiseDistribution.push_back(std::normal_distribution<double>(0., 1.));
        currentData_(i) = mean[i];
      }
      generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type OU"
                              " need to be vectors of same length");
    }
  }

  OU::OU(vector<double> mean, vector<double> theta, vector<double> phi){
    initParams(mean, theta, phi);
  }
  OU::OU(): OU({2., 4.3, 3., 0.5}, {1., 0.3, 2., 0.5}, {2., 2.1, 2.2, 2.3}){}

  OU::OU(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"mean", "theta", "phi"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config doesn't don't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
      vector<double> mean= std::any_cast<vector<double>>(params["mean"]);
      vector<double> theta= std::any_cast<vector<double>>(params["theta"]);
      vector<double> phi = std::any_cast<vector<double>>(params["phi"]);
      initParams(mean, theta, phi);
    }
    else{
      vector<double> mean{2., 4.3, 3., 0.5};
      vector<double> theta{1., 0.3, 2., 0.5};
      vector<double> phi{2., 2.1, 2.2, 2.3};
      initParams(mean, theta, phi);
    }
  }

  OU::OU(pybind11::dict py_config): OU::OU(makeConfigFromPyDict(py_config)){}

  const PriceVector& OU::getData() {
    for (int i=0; i < nAssets(); i++){
      double& x = currentData_(i);
      x += (theta[i]*(mean[i]-x)) + mean[i]*phi[i]*noiseDistribution[i](generator);
    }
    timestamp_ += 1;
    return currentData_;
  }

  // OU PAIR /////////////////////////////////////////////////////////////////////////
  void OUPair::initParams(double theta, double phi, double noise){
    this->theta=theta;
    this->phi=phi;
    ouNoiseDistribution = std::normal_distribution<double>(0., phi);
    randomWalkDistribution = std::normal_distribution<double>(0., noise);
    currentData_.resize(2.);
    for (int i=0; i < 2; i++){
      std::string assetName = "OUPair_" + std::to_string(i);
      this->assets_.push_back(Asset(assetName));
      currentData_(i) = 10.;
    }
    mean = 10;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  OUPair::OUPair(double theta, double phi, double noise){
    initParams(theta, phi, noise);
  }
  OUPair::OUPair(): OUPair(.015, .01, .03){}

  OUPair::OUPair(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"theta", "phi", "noise"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError("config for OUPair doesn't have all required constructor arguments");
      }
    }
    if (allKeysPresent){
      double theta = std::any_cast<double>(params["theta"]);
      double phi = std::any_cast<double>(params["phi"]);
      double noise = std::any_cast<double>(params["noise"]);
      initParams(theta, phi, noise);
    }
    else{
      double theta = .015;
      double phi = .01;
      double noise = .03;
      initParams(theta, phi, noise);
    }
  }

  OUPair::OUPair(pybind11::dict py_config): OUPair::OUPair(makeConfigFromPyDict(py_config)){}

  const PriceVector& OUPair::getData() {
    mean += mean*randomWalkDistribution(generator);
    currentData_(0) += (theta*(mean-currentData_(0))) + mean*ouNoiseDistribution(generator);
    currentData_(1) += (theta*(mean-currentData_(1))) + mean*ouNoiseDistribution(generator);
    // currentData_(0) = max(1., currentData_(0));
    // currentData_(0) = max(1., currentData_(0));
    timestamp_ += 1;
    return currentData_;
  }

  void OUPair::reset(){
    currentData_(0) = 10.;
    currentData_(1) = 10.;
    mean = 10;
  }

  // SIMPLE TREND //////////////////////////////////////////////////////////////////////
  void SimpleTrend::initParams(vector<double> trendProb, vector<int> minPeriod,
                               vector<int> maxPeriod, vector<double> noise,
                               vector<double> start, vector<double> dYMin,
                               vector<double> dYMax) {
    if ((trendProb.size() == minPeriod.size()) && (minPeriod.size() == maxPeriod.size())
        && (maxPeriod.size() == noise.size()) && (noise.size() == start.size())
        && (start.size() == dYMin.size()) && (dYMin.size() == dYMax.size())){
      this->trendProb=trendProb;
      this->minPeriod=minPeriod;
      this->maxPeriod=maxPeriod;
      this->noise=noise;
      this->dYMin=dYMin;
      this->dYMax=dYMax;
      this->start = start;
      currentData_.resize(trendProb.size());
      for (int i=0; i<trendProb.size(); i++){
        noiseDist.push_back(std::normal_distribution<double>(0., noise[i]));
        trendLenDist.push_back(std::uniform_int_distribution<int>
                                 (minPeriod[i], maxPeriod[i]));
        dYDist.push_back(std::uniform_real_distribution<double>
                                 (dYMin[i], dYMax[i]));
        trending.push_back(false);
        std::string assetName = "SimpleTrend_" + std::to_string(i);
        this->assets_.push_back(Asset(assetName));
        currentData_[i] = start[i]; // default starting val
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
        dY.push_back(0.);
      }
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type SimpleTrend"
                              " need to be vectors of same length");
    }
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  SimpleTrend::SimpleTrend(vector<double> trendProb, vector<int> minPeriod,
                           vector<int> maxPeriod, vector<double> noise,
                           vector<double> start, vector<double> dYMin,
                           vector<double> dYMax) {
    initParams(trendProb, minPeriod, maxPeriod, noise, start, dYMin, dYMax);
  }

  SimpleTrend::SimpleTrend(): SimpleTrend({0.001, 0.001}, {100, 500}, {200, 1500},
                                          {1. ,0.1}, {10., 15.},
                                          {0.001, 0.01}, {0.003, 0.03}){}

  SimpleTrend::SimpleTrend(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"trendProb", "minPeriod", "maxPeriod", "noise", "start",
                    "dYMin", "dYMax"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
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
    for (int i=0; i < nAssets(); i++){
      double& y = currentData_[i];
      if(trending[i]){
        y += y * dY[i] * currentDirection[i];
        if (--currentTrendLen[i] == 0){
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

  void SimpleTrend::reset(){
    for (int i=0; i<nAssets(); i++){
      currentData_[i] = start[i];
      trending[i] = false;
      currentDirection[i] = 1;
      currentTrendLen[i] = 0;
    }
  }


  // TREND OU //////////////////////////////////////////////////////////////////////

  void TrendOU::initParams(vector<double> trendProb, vector<int> minPeriod,
                           vector<int> maxPeriod, vector<double> dYMin,
                           vector<double> dYMax, vector<double> start,
                           vector<double> theta, vector<double> phi,
                           vector<double> noiseTrend, vector<double> emaAlpha){
    if ((trendProb.size() == minPeriod.size()) && (minPeriod.size() == maxPeriod.size())
        && (maxPeriod.size() == phi.size()) && (phi.size() == noiseTrend.size())
        && (start.size() == dYMin.size()) && (dYMin.size() == dYMax.size())
        && (dYMax.size() == theta.size()) && (theta.size() == phi.size())
        && (noiseTrend.size() == emaAlpha.size())){
      this->trendProb=trendProb;
      this->minPeriod=minPeriod;
      this->maxPeriod=maxPeriod;
      this->noiseTrend=noiseTrend;
      this->dYMin=dYMin;
      this->dYMax=dYMax;
      this->theta=theta;
      this->phi=phi;
      this->emaAlpha=emaAlpha;
      this->ema=start;
      this->ouMean=start;
      this->start=start;
      currentData_.resize(trendProb.size());
      for (int i=0; i<trendProb.size(); i++){
        ouNoiseDist.push_back(std::normal_distribution<double>(0., phi[i]));
        trendNoiseDist.push_back(std::normal_distribution<double>(0., noiseTrend[i]));
        trendLenDist.push_back(std::uniform_int_distribution<int>
                                 (minPeriod[i], maxPeriod[i]));
        dYDist.push_back(std::uniform_real_distribution<double>
                                 (dYMin[i], dYMax[i]));
        trending.push_back(false);
        std::string assetName = "TrendOU_" + std::to_string(i);
        this->assets_.push_back(Asset(assetName));
        currentData_ << start[i]; // default starting val
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
        dY.push_back(0.);
      }
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type TrendOU"
                              " need to be vectors of same length");
    }
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  TrendOU::TrendOU(vector<double> trendProb, vector<int> minPeriod,
                   vector<int> maxPeriod, vector<double> dYMin,
                   vector<double> dYMax, vector<double> start,
                   vector<double> theta, vector<double> phi,
                   vector<double> noiseTrend, vector<double> emaAlpha) {
    initParams(trendProb, minPeriod, maxPeriod, dYMin, dYMax, start,
               theta, phi, noiseTrend, emaAlpha);
  }

  TrendOU::TrendOU(): TrendOU({0.001, 0.001}, {100, 500},
                              {200, 1500}, {0.001, 0.01},
                              {0.003, 0.03}, {10., 15.},
                              {1., 0.5}, {2., 2.1},
                              {1., 1.2}, {0.1, 0.2}){}

  TrendOU::TrendOU(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"trendProb", "minPeriod", "maxPeriod", "dYMin",
                    "dYMax", "start", "theta", "phi", "noiseTrend", "emaAlpha"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError((string)"data_Source_config doesn't have all required"
                          "constructor arguments, missing: " + key );
      }
    }
    vector<double> trendProb = std::any_cast<vector<double>>(params["trendProb"]);
    vector<int> minPeriod = std::any_cast<vector<int>>(params["minPeriod"]);
    vector<int> maxPeriod = std::any_cast<vector<int>>(params["maxPeriod"]);
    vector<double> dYMin = std::any_cast<vector<double>>(params["dYMin"]);
    vector<double> dYMax = std::any_cast<vector<double>>(params["dYMax"]);
    vector<double> start = std::any_cast<vector<double>>(params["start"]);
    vector<double> theta = std::any_cast<vector<double>>(params["theta"]);
    vector<double> phi = std::any_cast<vector<double>>(params["phi"]);
    vector<double> noiseTrend = std::any_cast<vector<double>>(params["noiseTrend"]);
    vector<double> emaAlpha = std::any_cast<vector<double>>(params["emaAlpha"]);
    initParams(trendProb, minPeriod, maxPeriod, dYMin, dYMax, start,
               theta, phi, noiseTrend, emaAlpha);
  }

  TrendOU::TrendOU(pybind11::dict py_config):
    TrendOU::TrendOU(makeConfigFromPyDict(py_config)){}

  const PriceVector& TrendOU::getData(){
    for (int i=0; i < nAssets(); i++){
      double& y = currentData_[i];
      // TREND
      if(trending[i]){
        // double& x = currentData_(i);
        y += y * (dY[i] * currentDirection[i] + trendNoiseDist[i](generator));
        currentTrendLen[i] -= 1;
        if (currentTrendLen[i] == 0){
          trending[i] = false;
          ouMean[i] = y;
        }
        y = std::max(0.01, y);
        if (y <= .1){
          currentDirection[i] = 1;
        }
        // ema[i] += emaAlpha[i] * y + (1-emaAlpha[i]) * ema[i];
      }
      // OU
      else{
        double ou_noise = y*ouNoiseDist[i](generator); // relative to current y
        double ou_reverting_component = theta[i] * (ouMean[i] - y);
        // ou_process = theta[i] * (ema[i]-y) * dT + y * phi[i] * ouNoise[i](generator);
        y += ou_reverting_component + ou_noise;
        double rand = uniformDist(generator);
        if (rand < trendProb[i]){
          trending[i] = true;
          currentDirection[i] = (uniformDist(generator) < 0.5)? -1: 1;
          currentTrendLen[i] = trendLenDist[i](generator);
          dY[i] = dYDist[i](generator);
        }
      }
      // y += y*trendNoise[i](generator);
    }
    timestamp_ += 1;
    return currentData_;
  }

  void TrendOU::reset(){
    for (int i=0; i<trendProb.size(); i++){
      trending[i] = false;
      currentData_(i) = start[i];
      currentTrendLen[i] = 0;
      ouMean[i] = start[i];
    }
  }

  // TRENDY OU //////////////////////////////////////////////////////////////////////

  void TrendyOU::initParams(vector<double> trendProb, vector<int> minPeriod,
                            vector<int> maxPeriod, vector<double> dYMin,
                            vector<double> dYMax, vector<double> start,
                            vector<double> theta, vector<double> phi,
                            vector<double> noiseTrend, vector<double> emaAlpha){
    if ((trendProb.size() == minPeriod.size()) && (minPeriod.size() == maxPeriod.size())
        && (maxPeriod.size() == phi.size()) && (phi.size() == noiseTrend.size())
        && (start.size() == dYMin.size()) && (dYMin.size() == dYMax.size())
        && (dYMax.size() == theta.size()) && (theta.size() == phi.size())
        && (noiseTrend.size() == emaAlpha.size())){
      this->trendProb=trendProb;
      this->minPeriod=minPeriod;
      this->maxPeriod=maxPeriod;
      this->noiseTrend=noiseTrend;
      this->dYMin=dYMin;
      this->dYMax=dYMax;
      this->theta=theta;
      this->phi=phi;
      this->emaAlpha=emaAlpha;
      this->ema=start;
      this->start=start;
      // this->ouMean=start;
      currentData_.resize(trendProb.size());
      for (int i=0; i<trendProb.size(); i++){
        ouNoiseDist.push_back(std::normal_distribution<double>(0., phi[i]));
        trendNoiseDist.push_back(std::normal_distribution<double>(0., noiseTrend[i]));
        trendLenDist.push_back(std::uniform_int_distribution<int>
                                 (minPeriod[i], maxPeriod[i]));
        dYDist.push_back(std::uniform_real_distribution<double>
                                 (dYMin[i], dYMax[i]));
        ouMean.push_back(start[i]);
        ouComponent.push_back(0.);
        trendComponent.push_back(start[i]);
        trending.push_back(false);
        std::string assetName = "TrendyOU_" + std::to_string(i);
        this->assets_.push_back(Asset(assetName));
        currentData_ << start[i]; // default starting val
        currentDirection.push_back(1);
        currentTrendLen.push_back(0);
        dY.push_back(0.);
      }
    }
    else{
      throw std::length_error("parameters passed to DataSource<PriceVector> of type TrendyOU"
                              " need to be vectors of same length");
    }
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  }

  TrendyOU::TrendyOU(vector<double> trendProb, vector<int> minPeriod,
                     vector<int> maxPeriod, vector<double> dYMin,
                     vector<double> dYMax, vector<double> start,
                     vector<double> theta, vector<double> phi,
                     vector<double> noiseTrend, vector<double> emaAlpha) {
    initParams(trendProb, minPeriod, maxPeriod, dYMin, dYMax, start,
               theta, phi, noiseTrend, emaAlpha);
  }

  TrendyOU::TrendyOU(): TrendyOU({0.001, 0.001}, {100, 500},
                              {200, 1500}, {0.001, 0.01},
                              {0.003, 0.03}, {10., 15.},
                              {1., 0.5}, {2., 2.1},
                              {1., 1.2}, {0.1, 0.2}){}

  TrendyOU::TrendyOU(Config config){
    bool allKeysPresent{true};
    if (config.find("data_source_config") == config.end()){
      throw ConfigError("config passed but doesn't contain generator params");
      allKeysPresent = false;
    }
    Config params = std::any_cast<Config>(config["data_source_config"]);
    for (auto key: {"trendProb", "minPeriod", "maxPeriod", "dYMin",
                    "dYMax", "start", "theta", "phi", "noiseTrend", "emaAlpha"}){
      if (params.find(key) == params.end()){
        allKeysPresent=false;
        throw ConfigError((string)"data_Source_config doesn't have all required"
                          "constructor arguments, missing: " + key );
      }
    }
    vector<double> trendProb = std::any_cast<vector<double>>(params["trendProb"]);
    vector<int> minPeriod = std::any_cast<vector<int>>(params["minPeriod"]);
    vector<int> maxPeriod = std::any_cast<vector<int>>(params["maxPeriod"]);
    vector<double> dYMin = std::any_cast<vector<double>>(params["dYMin"]);
    vector<double> dYMax = std::any_cast<vector<double>>(params["dYMax"]);
    vector<double> start = std::any_cast<vector<double>>(params["start"]);
    vector<double> theta = std::any_cast<vector<double>>(params["theta"]);
    vector<double> phi = std::any_cast<vector<double>>(params["phi"]);
    vector<double> noiseTrend = std::any_cast<vector<double>>(params["noiseTrend"]);
    vector<double> emaAlpha = std::any_cast<vector<double>>(params["emaAlpha"]);
    initParams(trendProb, minPeriod, maxPeriod, dYMin, dYMax, start,
               theta, phi, noiseTrend, emaAlpha);
  }

  TrendyOU::TrendyOU(pybind11::dict py_config):
    TrendyOU::TrendyOU(makeConfigFromPyDict(py_config)){}

  const PriceVector& TrendyOU::getData(){
    for (int i=0; i < nAssets(); i++){
      double& y = currentData_[i];
      // OU
      // TREND
      // double trendNoise{0};
      double ouNoise = (trendComponent[i])*ouNoiseDist[i](generator);
      double ouRevertingComponent = theta[i] * (-ouComponent[i]); // centered at 0
      ouComponent[i] += ouRevertingComponent + ouNoise;
      if(trending[i]){
        // double& x = currentData_(i);
        // trendNoise =  trendNoiseDist[i](generator);
        trendComponent[i] += trendComponent[i] * (dY[i] * currentDirection[i]);// + trendNoiseDist[i](generator);
        trendComponent[i] = std::max(0.1, trendComponent[i]);
        // ouMean[i] = y;
        if (trendComponent[i] <= .1){
          currentDirection[i] = 1;
          trending[i] = true;
          currentTrendLen[i] = trendLenDist[i](generator);
        }
        if (--currentTrendLen[i] == 0){
          trending[i] = false;
          // ouMean[i] = y;
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
      y = ouComponent[i] + trendComponent[i];
      // ema[i] = emaAlpha[i] * ema[i] + (1-emaAlpha[i]) * y;
      // y = std::max(0.001, y);
    }
    timestamp_ += 1;
    return currentData_;
  }

  void TrendyOU::reset(){
    ema=start;
    ouMean=start;
    // this->ouMean=start;
    for (int i=0; i<trendProb.size(); i++){
      ouComponent[i] = 0;
      trendComponent[i] = start[i];
      trending[i] = false;
      currentData_(i) = start[i];
      currentTrendLen[i] = 0;
      // ema[i] = start[i];
      // ouMean[i] = start[i];
    }
  }



}// namespace madigan
