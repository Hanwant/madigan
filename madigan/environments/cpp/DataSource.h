#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <cmath>
#include <vector>
#include <exception>

#define PI2 (3.141592653589793238463*2)


namespace madigan{

  using std::vector;

  template<typename T>
  struct Data{
    Data();
    ~Data();
    T data;
  };


  template<typename T>
  class DataSource{
  public:
    DataSource();
    virtual ~DataSource()=0;
    // Data<T> nextData();
    T nextData();
  };


  class Synth: public DataSource<vector<double>>{
  public:
    int nAssets;
  public:
    Synth(double x, std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase);
    Synth(std::vector<double> freq, std::vector<double> mu,
          std::vector<double> amp, std::vector<double> phase);
    ~Synth();
    // Data<T> getData();
    vector<double> getData();

  private:
    vector<double> x;
    vector<double> freq;
    vector<double> mu;
    vector<double> amp;
    vector<double> initPhase;
  };

  Synth::Synth(double x, std::vector<double> freq, std::vector<double> mu,
               std::vector<double> amp, std::vector<double> phase): x(x){
    if (freq.size() == mu.size() && mu.size() == amp.size() &&
        amp.size() == phase.size()){
      this->freq=freq;
      this->mu=mu;
      this->amp=amp;
      this->initPhase=phase;
      this->x=phase;
      this->nAssets=freq.size();
    }
    else{
      throw std::length_error("parameters passed to DataSource of type Synth"
                              " need to be vectors of same length");
    }
  }

  Synth::Synth(std::vector<double> freq, std::vector<double> mu,
               std::vector<double> amp, std::vector<double> phase): x(x){
    Synth(0., freq, mu, amp, phase);
  }

  std::vector<double> Synth::getData(){
    std::vector<double> out(nAssets);
    for (int i=0; i < nAssets; i++){
      out[i] = mu[i] + amp[i] * std::sin(PI2*x[i]*freq[i]);
    }
    return out;

  }

} // namespace madigan


#endif 
