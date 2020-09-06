#include <vector>
#include <math.h>

#define PI 3.14159265

class Generator{
public:
  virtual std::vector<double> next()=0;
  virtual ~Generator(){}
};


class SineGenerator: public Generator{
public:
  std::vector<double> m_freq;
  std::vector<double> m_mu;
  std::vector<double> m_amp;
  std::vector<double> m_phase;
  double dx;
  int nEle;
  double m_x=0.;

public:
  SineGenerator(): m_freq{1., 2., 3., 4.}, m_mu{2., 3., 4., 5.},
                   m_amp{1., 2., 3., 4.}, m_phase{0., 1., 2., 0.},
                   dx(0.01), nEle(4){
  }
  SineGenerator(std::vector<double> freq, std::vector<double> mu,
                std::vector<double> amp, std::vector<double> phase,
                double dx): dx(dx){
    if(freq.size() != mu.size() || mu.size()!= amp.size()
       || amp.size() != phase.size()){
      throw "Generator Parameters must all be of the same length";
    }
    m_freq=freq;
    m_mu=mu;
    m_amp=amp;
    m_phase=phase;
    nEle = freq.size();
  }

  std::vector<double> next() override {
    std::vector<double> out;
    for(int i=0; i!=nEle; i++){
      out.push_back(m_mu[i]+m_amp[i]*std::sin(2*PI*m_x*m_freq[i]));
    }
        m_x += dx;
    return out;
  }
};
