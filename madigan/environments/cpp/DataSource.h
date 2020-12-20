#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>
#include <cstdint>
#include <random>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Assets.h"
#include "DataTypes.h"
#include "Config.h"
#include "randomBoolGenerator.h"
#include "WaveTableOsc.h"
// #include "Portfolio.h"

#define PI2 (3.141592653589793238463*2)


namespace madigan{

  using std::vector;
  using std::size_t;
  using randomBoolGenerator = XorShift128PlusBitShifterPseudoRandomBooleanGenerator;

  template<class T>
  class DataSource{
  public:
    Assets assets;
    int nAssets_;
  public:
    // virtual ~DataSource(){}
    int nAssets() const;
    const T& getData();
    const T& currentData() const;
    const T& currentPrices() const;
    void reset();
    std::size_t currentTime() const;
  };

  template<>
  class DataSource<PriceVector>{
  public:
    Assets assets;
    int nAssets_;
  public:
    int nAssets() const{ return nAssets_;}

    virtual const PriceVector& getData()=0;
    virtual const PriceVector& currentData() const=0;
    virtual const PriceVector& currentPrices() const=0;
    virtual void reset()=0;
    virtual std::size_t currentTime() const =0;
  };

  template<>
  class DataSource<PriceMatrix>{
  public:
    Assets assets;
    int nAssets_;
  public:
    int nAssets() const{ return nAssets_;}
    virtual const PriceMatrix& getData()=0;
    virtual const PriceMatrix& currentData() const=0;
    virtual const PriceMatrix& currentPrices() const=0;
    virtual void reset()=0;
    virtual std::size_t currentTime() const =0;
  };

  using DataSourceBidAsk = DataSource<PriceMatrix>;
  using DataSourceTick = DataSource<PriceVector>;

  template<class T>
  std::unique_ptr<DataSource<T>> makeDataSource(string dataSourceType);
  template<class T>
  std::unique_ptr<DataSource<T>> makeDataSource(string dataSourceType, Config config);


  // The following DataSources load data from files
  class HDFSource: public DataSourceTick{
  public:
    string filepath;
    string mainKey;
    string timestampKey;
    string priceKey;
    int nAssets_{1};
  public:
    HDFSource(string datapath, string mainKey,
              string pricekey, string timestampKey);
    HDFSource(Config config);
    HDFSource(pybind11::dict config): HDFSource(makeConfigFromPyDict(config)){}
    void loadData();
    int nAssets() const{ return nAssets_;}
    const PriceVector& getData();
    const PriceVector& currentData() const{return currentPrices_;}
    const PriceVector& currentPrices() const{return currentPrices_;}
    void reset(){}
    int size(){ return prices_.size();}
    std::size_t currentTime() const{return timestamp_;}

  private:
    PriceVector prices_;
    PriceVector currentPrices_{1};
    TimeVector timestamps_;
    std::size_t timestamp_;
    int currentIdx_{0};
  };


  // SYNTHS - The Following DataSourceTick<PriceVector>s are Synthetic Time Series
  // Composite can combine outputs of many different data sources
  class Composite: public DataSourceTick{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    Composite()=delete;
    Composite(Config config);
    Composite(pybind11::dict config):
      Composite(makeConfigFromPyDict(config)){}
    const PriceVector& getData();
    const PriceVector& currentData() const{return currentData_;}
    const PriceVector& currentPrices() const{return currentData_;}
    std::size_t currentTime() const{return timestamp_;}
    const vector<std::unique_ptr<DataSourceTick>>& dataSources() const{ return dataSources_;}
    void reset() {for (auto& source: dataSources_){
        source->reset();
      }}
  private:
    vector<std::unique_ptr<DataSourceTick>> dataSources_;
    PriceVector currentPrices_;
    PriceVector currentData_;
    std::size_t timestamp_;

  };

  // Base for Periodic Wave funcitons I.e sine, triangle, sawtooth
  class Synth: public DataSourceTick{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    Synth(); // use default values for parameters
    Synth(std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase,
          double dX): Synth(freq, mu, amp, phase, dX, 0.){}
    Synth(std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase,
          double dX, double noise);
    Synth(Config config);
    Synth(pybind11::dict config);
    ~Synth(){}
    const PriceVector& getData() ;
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<double> freq, std::vector<double> mu,
                            std::vector<double> amp, std::vector<double> phase,
                            double dX, double noise);

  protected:
    double dX{0.01};
    double noise{0.};
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> initPhase;
    vector<double> x;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::normal_distribution<double> noiseDistribution;
    PriceVector currentData_;
  };

  class SineAdder: public DataSourceTick{
  public:
    SineAdder(); // use default values for parameters
    SineAdder(std::vector<double> freq, std::vector<double> mu,
              std::vector<double> amp, std::vector<double> phase,
              double dX): SineAdder(freq, mu, amp, phase, dX, 0.){}
    SineAdder(std::vector<double> freq, std::vector<double> mu,
              std::vector<double> amp, std::vector<double> phase,
              double dX, double noise);
    SineAdder(Config config);
    SineAdder(pybind11::dict config);
    int nAssets() const{ return nAssets_;}
    ~SineAdder(){}
    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }
  protected:
    void initParams(std::vector<double> freq, std::vector<double> mu,
                    std::vector<double> amp, std::vector<double> phase,
                    double dX, double noise);
  protected:
    double dX{0.01};
    double noise{0.};
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> initPhase;
    vector<double> x;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::normal_distribution<double> noiseDistribution;
    PriceVector currentData_;
  };

  class SineDynamic: public DataSourceTick{
  public:
    SineDynamic(); // use default values for parameters
    SineDynamic(std::vector<std::array<double, 3>> freqRange,
                std::vector<std::array<double, 3>> muRange,
                std::vector<std::array<double, 3>> ampRange,
                std::vector<double> phase,
                double dX): SineDynamic(freqRange, muRange, ampRange, phase, dX, 0.){}
    SineDynamic(std::vector<std::array<double, 3>> freqRange,
                std::vector<std::array<double, 3>> muRange,
                std::vector<std::array<double, 3>> ampRange,
                std::vector<double> phase,
              double dX, double noise);
    SineDynamic(Config config);
    SineDynamic(pybind11::dict config);
    int nAssets() const{ return nAssets_;}
    ~SineDynamic(){}
    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    double getProcess(int i);
    void reset();
    std::size_t currentTime() const { return timestamp_; }
  protected:
    void initParams(std::vector<std::array<double, 3>> freq,
                    std::vector<std::array<double, 3>> mu,
                    std::vector<std::array<double, 3>> amp,
                    std::vector<double> phase,
                    double dX, double noise);
    void updateParams();

  protected:
    double dX = 0;
    int sampleRate;
    double noise{0.};
    int nComponents;
    int stepsSinceUpdate=0;
    vector<WaveTableOsc<double>> oscillators;
    // double, double, double == low, high, dt
    vector<std::array<double, 3>> freqRange;
    vector<std::array<double, 3>> muRange;
    vector<std::array<double, 3>> ampRange;
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> initPhase;
    vector<double> x;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::normal_distribution<double> noiseDistribution;
    std::uniform_real_distribution<double> updateParameterDist;
    vector<std::uniform_real_distribution<double>> freqDist;
    vector<std::uniform_real_distribution<double>> muDist;
    vector<std::uniform_real_distribution<double>> ampDist;
    randomBoolGenerator boolDist;
    PriceVector currentData_;
  };

  class SawTooth: public Synth{
  public:
    using Synth::Synth;
    const PriceVector& getData();
  };

  class Triangle: public Synth{
  public:
    using Synth::Synth;
    const PriceVector& getData();
  };

  class OU: public DataSourceTick{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    OU();
    OU(std::vector<double> mean, std::vector<double> theta,
       std::vector<double> phi, std::vector<double> noise_var);
    OU(Config config);
    OU(pybind11::dict config);
    ~OU(){}
    const PriceVector& getData();
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<double> mean, std::vector<double> theta,
                            std::vector<double> phi, std::vector<double> noise_var);

  protected:
    const double dT{1.};
    vector<double> mean;
    vector<double> theta;
    vector<double> phi;
    vector<double> noise_var;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> noiseDistribution;
    PriceVector currentData_;
  };

  class OUDynamic: public DataSourceTick{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    OUDynamic();
    OUDynamic(std::vector<std::array<double, 3>> meanRange,
              std::vector<std::array<double, 3>> thetaRange,
              std::vector<std::array<double, 3>> phiRange);
    OUDynamic(Config config);
    OUDynamic(pybind11::dict config);
    ~OUDynamic(){}
    const PriceVector& getData();
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<std::array<double, 3>> meanRange,
                            std::vector<std::array<double, 3>> thetaRange,
                            std::vector<std::array<double, 3>> phiRange);

  protected:
    const double dT{1.};
    vector<double> mean;
    vector<double> theta;
    vector<double> phi;
    vector<std::array<double, 3>> meanRange;
    vector<std::array<double, 3>> thetaRange;
    vector<std::array<double, 3>> phiRange;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> noiseDistribution;
    randomBoolGenerator boolDist;
    PriceVector currentData_;
  };

  class SimpleTrend: public DataSourceTick{
  public:
    SimpleTrend();
    SimpleTrend(std::vector<double> trendProb, std::vector<int> minPeriod,
                std::vector<int> maxPeriod, std::vector<double> noise,
                std::vector<double> dYMin, std::vector<double> dYMax,
                std::vector<double> start);
    SimpleTrend(Config config);
    SimpleTrend(pybind11::dict config);
    ~SimpleTrend(){}

    // int nAssets() const { return nAssets_;}
    const PriceVector& getData();
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                             std::vector<int> maxPeriod, std::vector<double> noise,
                            std::vector<double> dYMin, std::vector<double> dYMax,
                            std::vector<double> start);

  protected:
    const double dT{1.};
    std::vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> noise;
    vector<double> dY;
    vector<double> dYMin;
    vector<double> dYMax;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> noiseDist;
    std::vector<std::uniform_real_distribution<double>> dYDist;
    std::vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;

    std::vector<bool> trending;
    std::vector<int> currentDirection;
    std::vector<int> currentTrendLen;

  };

  class TrendOU: public DataSourceTick{
  public:
    TrendOU();
    TrendOU(std::vector<double> trendProb, std::vector<int> minPeriod,
            std::vector<int> maxPeriod, std::vector<double> dYMin,
            std::vector<double> dYMax, std::vector<double> start,
            std::vector<double> theta, std::vector<double> phi,
            std::vector<double> noise_var, std::vector<double> emaAlpha);
    TrendOU(Config config);
    TrendOU(pybind11::dict config);
    ~TrendOU(){}

    const virtual PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset();
    std::size_t currentTime() const { return timestamp_; }

  private:
    void initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                    std::vector<int> maxPeriod, std::vector<double> dYMin,
                    std::vector<double> dYMax, std::vector<double> start,
                    std::vector<double> theta, std::vector<double> phi,
                    std::vector<double> noise_var, std::vector<double> emaAlpha);
  private:
    const double dT{1.};
    std::vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> dY;
    vector<double> dYMin;
    vector<double> dYMax;
    vector<double> theta;
    vector<double> phi;
    vector<double> noiseTrend;
    vector<double> emaAlpha;
    vector<double> ema;
    vector<double> start; // for resetting
    std::vector<double> ouMean;
    std::size_t timestamp_{0};
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> ouNoiseDist;
    std::vector<std::normal_distribution<double>> trendNoiseDist;
    std::vector<std::uniform_real_distribution<double>> dYDist;
    std::vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;


    std::vector<bool> trending;
    std::vector<int> currentDirection;
    std::vector<int> currentTrendLen;

  };

  class TrendyOU: public DataSourceTick{
  public:
    TrendyOU();
    TrendyOU(std::vector<double> trendProb, std::vector<int> minPeriod,
            std::vector<int> maxPeriod, std::vector<double> dYMin,
            std::vector<double> dYMax, std::vector<double> start,
            std::vector<double> theta, std::vector<double> phi,
            std::vector<double> noise_var, std::vector<double> emaAlpha);
    TrendyOU(Config config);
    TrendyOU(pybind11::dict config);
    ~TrendyOU(){}

    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset();
    std::size_t currentTime() const { return timestamp_; }

  private:
    void initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                    std::vector<int> maxPeriod, std::vector<double> dYMin,
                    std::vector<double> dYMax, std::vector<double> start,
                    std::vector<double> theta, std::vector<double> phi,
                    std::vector<double> noise_var, std::vector<double> emaAlpha);
  private:
    const double dT{1.};
    std::vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> dY;
    vector<double> dYMin;
    vector<double> dYMax;
    vector<double> theta;
    vector<double> phi;
    vector<double> noiseTrend;
    vector<double> emaAlpha;
    vector<double> ema;
    vector<double> start;
    vector<double> ouComponent;
    vector<double> trendComponent;
    std::vector<double> ouMean;
    std::size_t timestamp_{0};
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> ouNoiseDist;
    std::vector<std::normal_distribution<double>> trendNoiseDist;
    std::vector<std::uniform_real_distribution<double>> dYDist;
    std::vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;


    std::vector<bool> trending;
    std::vector<int> currentDirection;
    std::vector<int> currentTrendLen;

  };

} // namespace madigan


#endif
