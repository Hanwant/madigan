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
    int nAssets() const{ return assets_.size();}
    Assets assets() const { return assets_; }

    virtual const PriceVector& getData()=0;
    virtual const PriceVector& currentData() const=0;
    virtual const PriceVector& currentPrices() const=0;
    virtual void reset()=0;
    virtual std::size_t currentTime() const =0;
  protected:
    Assets assets_;
  };

  template<>
  class DataSource<PriceMatrix>{
  public:
    int nAssets() const{ return assets_.size();}
    Assets assets() const {return assets_; }
    virtual const PriceMatrix& getData()=0;
    virtual const PriceMatrix& currentData() const=0;
    virtual const PriceMatrix& currentPrices() const=0;
    virtual void reset()=0;
    virtual std::size_t currentTime() const =0;
  protected:
    Assets assets_;
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

  public:
    HDFSource(string datapath, string mainKey,
              string pricekey, string timestampKey);
    HDFSource(Config config);
    HDFSource(pybind11::dict config): HDFSource(makeConfigFromPyDict(config)){}
    void loadData();
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
    Synth(); // use default values for parameters
    Synth(vector<double> freq, vector<double> mu,
          vector<double> amp, vector<double> phase,
          double dX): Synth(freq, mu, amp, phase, dX, 0.){}
    Synth(vector<double> freq, vector<double> mu,
          vector<double> amp, vector<double> phase,
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
    virtual void initParams(vector<double> freq, vector<double> mu,
                            vector<double> amp, vector<double> phase,
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
    SineAdder(vector<double> freq, vector<double> mu,
              vector<double> amp, vector<double> phase,
              double dX): SineAdder(freq, mu, amp, phase, dX, 0.){}
    SineAdder(vector<double> freq, vector<double> mu,
              vector<double> amp, vector<double> phase,
              double dX, double noise);
    SineAdder(Config config);
    SineAdder(pybind11::dict config);
    ~SineAdder(){}
    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }
  protected:
    void initParams(vector<double> freq, vector<double> mu,
                    vector<double> amp, vector<double> phase,
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
    SineDynamic(vector<std::array<double, 3>> freqRange,
                vector<std::array<double, 3>> muRange,
                vector<std::array<double, 3>> ampRange,
                double dX): SineDynamic(freqRange, muRange, ampRange, dX, 0.){}
    SineDynamic(vector<std::array<double, 3>> freqRange,
                vector<std::array<double, 3>> muRange,
                vector<std::array<double, 3>> ampRange,
                double dX, double noise);
    SineDynamic(Config config);
    SineDynamic(pybind11::dict config);
    ~SineDynamic(){}
    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    double getProcess(int i);
    void reset();
    std::size_t currentTime() const { return timestamp_; }
  protected:
    void initParams(vector<std::array<double, 3>> freq,
                    vector<std::array<double, 3>> mu,
                    vector<std::array<double, 3>> amp,
                    double dX, double noise);
    void updateParams();

  protected:
    double dX = 0;
    int sampleRate;
    double noise{0.};
    int nComponents;
    int stepsSinceUpdate=0;
    vector<WaveTableOsc<double>> oscillators;
    vector<std::array<double, 3>> freqRange;
    vector<std::array<double, 3>> muRange;
    vector<std::array<double, 3>> ampRange;
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
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

  class SineDynamicTrend: public DataSourceTick{
  public:
    SineDynamicTrend(); // use default values for parameters
    SineDynamicTrend(vector<std::array<double, 3>> freqRange,
                     vector<std::array<double, 3>> muRange,
                     vector<std::array<double, 3>> ampRange,
                     vector<std::array<int, 2>> trendRange,
                     vector<double> trendIncr, vector<double> trendProb,
                     double dX): SineDynamicTrend(freqRange, muRange,
                                                  ampRange, trendRange,
                                                  trendIncr, trendProb, dX, 0.){}
    SineDynamicTrend(vector<std::array<double, 3>> freqRange,
                     vector<std::array<double, 3>> muRange,
                     vector<std::array<double, 3>> ampRange,
                     vector<std::array<int, 2>> trendRange,
                     vector<double> trendIncr, vector<double> trendProb,
                     double dX, double noise);
    SineDynamicTrend(Config config);
    SineDynamicTrend(pybind11::dict config);
    ~SineDynamicTrend(){}
    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    double getProcess(int i);
    void reset();
    std::size_t currentTime() const { return timestamp_; }
  protected:
    void initParams(vector<std::array<double, 3>> freqRange,
                    vector<std::array<double, 3>> muRange,
                    vector<std::array<double, 3>> ampRange,
                    vector<std::array<int, 2>> trendRange,
                    vector<double> trendIncr, vector<double> trendProb,
                    double dX, double noise);
    void updateParams();

  protected:
    double dX = 0;
    int sampleRate;
    double noise{0.};
    int nComponents;
    int stepsSinceUpdate=0;
    vector<WaveTableOsc<double>> oscillators;
    vector<std::array<double, 3>> freqRange;
    vector<std::array<double, 3>> muRange;
    vector<std::array<double, 3>> ampRange;
    vector<std::array<int, 2>> trendRange;
    vector<double> trendIncr;
    vector<double> trendProb;
    vector<bool> trending;
    vector<int> currentDirection;
    vector<int> currentTrendLen;
    double trendComponent;
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> x;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::normal_distribution<double> noiseDistribution;
    std::uniform_real_distribution<double> updateParameterDist;
    vector<std::uniform_real_distribution<double>> freqDist;
    vector<std::uniform_real_distribution<double>> muDist;
    vector<std::uniform_real_distribution<double>> ampDist;
    std::uniform_real_distribution<double> trendProbDist;
    vector<std::uniform_int_distribution<int>> trendLenDist;
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

  class Gaussian: public DataSourceTick{
  public:
    Gaussian();
    Gaussian(vector<double> mean, vector<double> var);
    Gaussian(Config config);
    Gaussian(pybind11::dict config);
    ~Gaussian(){}
    const PriceVector& getData();
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset(){}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(vector<double> mean, vector<double> var);

  protected:
    const double dT{1.};
    vector<double> mean;
    vector<double> var;
    std::size_t timestamp_;
    std::default_random_engine generator;
    vector<std::normal_distribution<double>> noiseDistribution;
    PriceVector currentData_;
  };

  class OU: public DataSourceTick{
  public:
    OU();
    OU(vector<double> mean, vector<double> theta,
       vector<double> phi);
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
    virtual void initParams(vector<double> mean, vector<double> theta,
                            vector<double> phi);

  protected:
    const double dT{1.};
    vector<double> mean;
    vector<double> theta;
    vector<double> phi;
    std::size_t timestamp_;
    std::default_random_engine generator;
    vector<std::normal_distribution<double>> noiseDistribution;
    PriceVector currentData_;
  };

  class OUDynamic: public DataSourceTick{
  public:
    OUDynamic();
    OUDynamic(vector<std::array<double, 3>> meanRange,
              vector<std::array<double, 3>> thetaRange,
              vector<std::array<double, 3>> phiRange);
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
    virtual void initParams(vector<std::array<double, 3>> meanRange,
                            vector<std::array<double, 3>> thetaRange,
                            vector<std::array<double, 3>> phiRange);

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
    vector<std::normal_distribution<double>> noiseDistribution;
    randomBoolGenerator boolDist;
    PriceVector currentData_;
  };

  class SimpleTrend: public DataSourceTick{
  public:
    SimpleTrend();
    SimpleTrend(vector<double> trendProb, vector<int> minPeriod,
                vector<int> maxPeriod, vector<double> noise,
                vector<double> dYMin, vector<double> dYMax,
                vector<double> start);
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
    virtual void initParams(vector<double> trendProb, vector<int> minPeriod,
                            vector<int> maxPeriod, vector<double> noise,
                            vector<double> dYMin, vector<double> dYMax,
                            vector<double> start);

  protected:
    const double dT{1.};
    vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> noise;
    vector<double> dY;
    vector<double> dYMin;
    vector<double> dYMax;
    std::size_t timestamp_;
    std::default_random_engine generator;
    vector<std::normal_distribution<double>> noiseDist;
    vector<std::uniform_real_distribution<double>> dYDist;
    vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;

    vector<bool> trending;
    vector<int> currentDirection;
    vector<int> currentTrendLen;

  };

  class TrendOU: public DataSourceTick{
  public:
    TrendOU();
    TrendOU(vector<double> trendProb, vector<int> minPeriod,
            vector<int> maxPeriod, vector<double> dYMin,
            vector<double> dYMax, vector<double> start,
            vector<double> theta, vector<double> phi,
            vector<double> noise_var, vector<double> emaAlpha);
    TrendOU(Config config);
    TrendOU(pybind11::dict config);
    ~TrendOU(){}

    const virtual PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset();
    std::size_t currentTime() const { return timestamp_; }

  private:
    void initParams(vector<double> trendProb, vector<int> minPeriod,
                    vector<int> maxPeriod, vector<double> dYMin,
                    vector<double> dYMax, vector<double> start,
                    vector<double> theta, vector<double> phi,
                    vector<double> noise_var, vector<double> emaAlpha);
  private:
    const double dT{1.};
    vector<double> trendProb;
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
    vector<double> ouMean;
    std::size_t timestamp_{0};
    std::default_random_engine generator;
    vector<std::normal_distribution<double>> ouNoiseDist;
    vector<std::normal_distribution<double>> trendNoiseDist;
    vector<std::uniform_real_distribution<double>> dYDist;
    vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;


    vector<bool> trending;
    vector<int> currentDirection;
    vector<int> currentTrendLen;

  };

  class TrendyOU: public DataSourceTick{
  public:
    TrendyOU();
    TrendyOU(vector<double> trendProb, vector<int> minPeriod,
            vector<int> maxPeriod, vector<double> dYMin,
            vector<double> dYMax, vector<double> start,
            vector<double> theta, vector<double> phi,
            vector<double> noise_var, vector<double> emaAlpha);
    TrendyOU(Config config);
    TrendyOU(pybind11::dict config);
    ~TrendyOU(){}

    const PriceVector& getData() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    void reset();
    std::size_t currentTime() const { return timestamp_; }

  private:
    void initParams(vector<double> trendProb, vector<int> minPeriod,
                    vector<int> maxPeriod, vector<double> dYMin,
                    vector<double> dYMax, vector<double> start,
                    vector<double> theta, vector<double> phi,
                    vector<double> noise_var, vector<double> emaAlpha);
  private:
    const double dT{1.};
    vector<double> trendProb;
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
    vector<double> ouMean;
    std::size_t timestamp_{0};
    std::default_random_engine generator;
    vector<std::normal_distribution<double>> ouNoiseDist;
    vector<std::normal_distribution<double>> trendNoiseDist;
    vector<std::uniform_real_distribution<double>> dYDist;
    vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;


    vector<bool> trending;
    vector<int> currentDirection;
    vector<int> currentTrendLen;

  };

} // namespace madigan


#endif
