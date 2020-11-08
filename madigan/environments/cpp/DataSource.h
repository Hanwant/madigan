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
    int nAssets();
    virtual const PriceVector& getData()=0;
    virtual const PriceVector& currentData() const=0;
    virtual const PriceVector& currentPrices() const=0;
    virtual std::size_t currentTime() const =0;
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
    int nAssets() const { return nAssets_;}
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
  class SineAdder: public Synth{
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
    ~SineAdder(){}
    const PriceVector& getData() override;
  protected:
    void initParams(std::vector<double> _freq, std::vector<double> _mu,
                               std::vector<double> _amp, std::vector<double> _phase,
                               double _dX, double _noise) override;
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
    int nAssets() const { return nAssets_;}
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
                std::vector<double> dY, std::vector<double> start);
    SimpleTrend(Config config);
    SimpleTrend(pybind11::dict config);
    ~SimpleTrend(){}

    int nAssets() const { return nAssets_;}
    const PriceVector& getData() override;
    const PriceVector& currentData() const{ return currentData_;}
    const PriceVector& currentPrices() const{ return currentData_;}
    std::size_t currentTime() const { return timestamp_; }

  protected:
    virtual void initParams(std::vector<double> trendProb, std::vector<int> minPeriod,
                             std::vector<int> maxPeriod, std::vector<double> noise,
                             std::vector<double> dY, std::vector<double> start);

  protected:
    const double dT{1.};
    std::vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> noise;
    vector<double> dY;
    std::size_t timestamp_;
    std::default_random_engine generator;
    std::vector<std::uniform_int_distribution<int>> trendLenPicker;
    std::vector<std::normal_distribution<double>> noiseDist;
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

} // namespace oadigan


#endif
