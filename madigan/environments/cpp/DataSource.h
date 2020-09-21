#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>
#include <cstdint>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <pybind11/eigen.h>

#include "Assets.h"
#include "DataTypes.h"
// #include "Portfolio.h"

#define PI2 (3.141592653589793238463*2)


namespace madigan{

  using std::vector;


  class DataSource{
  public:
    int nAssets;
    Assets assets;
  public:
    // DataSource(const DataSource&) =delete;
    // DataSource& operator=(const DataSource&) =delete;
    virtual ~DataSource(){}
    // Data<T> nextData();
    virtual const PriceVector& getData()=0;
    virtual const PriceVector& currentData() const=0;
  };


  class Synth: public DataSource{
  public:
    int nAssets;
    Assets assets;
  public:
    Synth(); // use default values for parameters
    Synth(std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase,
          double dX);
    ~Synth(){}
    // Data<T> getData();
    const PriceVector& getData() override;
    const pybind11::array_t<double> getData_np() ;
    const PriceVector& currentData() const{ return currentData_;}

  private:
    void initParams(std::vector<double> freq, std::vector<double> mu,
                    std::vector<double> amp, std::vector<double> phase,
                    double dX);

  private:
    double dX;
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> initPhase;
    vector<double> x;

    PriceVector currentData_;
  };

} // namespace oadigan


#endif
