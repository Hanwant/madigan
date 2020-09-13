#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define PI2 (3.141592653589793238463*2)


namespace madigan{

  using std::vector;

  template<typename T>
  struct Data{
    Data();
    ~Data();
    T data;
  };

  // typedef Data<std::pair<std::uint64_t, vector<double>>> PriceItem;
  typedef vector<double> PriceVector;

  class DataSource{
  public:
    virtual ~DataSource(){}
    // Data<T> nextData();
    virtual const PriceVector &getData()=0;
  };


  class Synth: public DataSource{
  public:
    int nAssets;
  public:
    Synth(); // use default values for parameters
    Synth(std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase,
          double dX);
    ~Synth(){}
    // Data<T> getData();
    const PriceVector& getData() override;
    const pybind11::array_t<double> getData_np() ;

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

    PriceVector currentData;
  };

} // namespace madigan


#endif
