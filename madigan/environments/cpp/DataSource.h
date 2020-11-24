#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>
#include <cstdint>
#include <random>
#include <chrono>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Assets.h"
#include "DataTypes.h"
#include "Config.h"
// #include "Portfolio.h"

#define PI2 (3.141592653589793238463*2)


namespace madigan{

  using std::vector;


  class DataSource{
  public:
    Assets assets;
    int nAssets_;
  public:
    // DataSource(const DataSource&) =delete;
    // DataSource& operator=(const DataSource&) =delete;
    virtual ~DataSource(){}
    // Data<T> nextData();
    int nAssets() const{ return nAssets_;}
    virtual const PriceVector& getData()=0;
    virtual const PriceVector& currentData() const=0;
    virtual const PriceVector& currentPrices() const=0;
    virtual std::size_t currentTime() const =0;
  };

  // Composite can combine outputs of many different data sources
  class Composite: public DataSource{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    Composite()=delete;
    Composite(Config config);
    Composite(pybind11::dict config){
      Composite(makeConfigFromPyDict(config));}
    // int nAssets() const{ return nAssets_;}
    const PriceVector& getData();
    const PriceVector& currentData() const{return currentData_;}
    const PriceVector& currentPrices() const{return currentData_;}
    std::size_t currentTime() const{return timestamp_;}
    const vector<std::unique_ptr<DataSource>>& dataSources() const{ return dataSources_;}
  private:
    vector<std::unique_ptr<DataSource>> dataSources_;
    PriceVector currentPrices_;
    PriceVector currentData_;
    std::size_t timestamp_;

  };


  class Synth: public DataSource{
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
    // Data<T> getData();
    // int nAssets() const { return nAssets_;}
    const PriceVector& getData() override;
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
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
  class SineAdder: public DataSource{
  public:
    // using Synth::Synth;
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
    const PriceVector& getData() override;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
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
  class SawTooth: public Synth{
  public:
    using Synth::Synth;
    const PriceVector& getData() override;
  };
  class Triangle: public Synth{
  public:
    using Synth::Synth;
    const PriceVector& getData() override;
  };

  class OU: public DataSource{
  public:
    Assets assets;
    int nAssets_{0};
  public:
    OU(); // use default values for parameters
    OU(std::vector<double> mean, std::vector<double> theta,
       std::vector<double> phi, std::vector<double> noise_var);
    OU(Config config);
    OU(pybind11::dict config);
    ~OU(){}
    // Data<T> getData();
    // int nAssets() const { return nAssets_;}
    const PriceVector& getData() override;
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
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

  class SimpleTrend: public DataSource{
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
    const PriceVector& getData() override;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
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

  class TrendOU: public DataSource{
  public:
    TrendOU();
    TrendOU(std::vector<double> trendProb, std::vector<int> minPeriod,
            std::vector<int> maxPeriod, std::vector<double> noise,
            std::vector<double> dYMin, std::vector<double> dYMax,
            std::vector<double> start, std::vector<double> theta,
            std::vector<double> phi, std::vector<double> noise_var,
            std::vector<double> emaAlpha);
    TrendOU(Config config);
    TrendOU(pybind11::dict config);
    ~TrendOU(){}

    // int nAssets() const { return nAssets_;}
    const PriceVector& getData() override;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                            std::vector<int> maxPeriod, std::vector<double> noise,
                            std::vector<double> dYMin, std::vector<double> dYMax,
                            std::vector<double> start, std::vector<double> theta,
                            std::vector<double> phi, std::vector<double> noise_var,
                            std::vector<double> emaAlpha);
  protected:
    const double dT{1.};
    std::vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> noise;
    vector<double> dY;
    vector<double> dYMin;
    vector<double> dYMax;
    vector<double> theta;
    vector<double> phi;
    vector<double> noise_var;
    vector<double> emaAlpha;
    vector<double> ema;
    std::size_t timestamp_{0};
    std::default_random_engine generator;
    std::vector<std::normal_distribution<double>> trendNoise;
    std::vector<std::normal_distribution<double>> ouNoise;
    std::vector<std::uniform_real_distribution<double>> dYDist;
    std::vector<std::uniform_int_distribution<int>> trendLenDist;
    std::uniform_real_distribution<double> uniformDist{0., 1.};
    PriceVector currentData_;

    std::vector<bool> trending;
    std::vector<int> currentDirection;
    std::vector<int> currentTrendLen;

  };

  std::unique_ptr<DataSource> makeDataSource(string dataSourceType);
  std::unique_ptr<DataSource> makeDataSource(string dataSourceType, Config config);


  // class PySynth: public Synth{
  // public:
  //   using Synth::Synth;

  //   const PriceVector& getData() override {
  //     PYBIND11_OVERLOAD(const PriceVector&,
  //                       Synth,
  //                       getData, );
  //   }

} // namespace madigan


#endif
