#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>

#define PI2 (3.141592653589793238463*2)


namespace madigan{

using std::vector;

template<typename T>
struct Data{
  Data();
  ~Data();
  T data;
};


class DataSource{
public:
  virtual ~DataSource(){}
  // Data<T> nextData();
  virtual vector<double> getData()=0;
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
  vector<double> getData() override;

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
};

} // namespace madigan


#endif 
