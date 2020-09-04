#include <vector>
#include <math.h>

#define PI 3.14159265

class Generator{
public:
  virtual std::vector<float> next()=0;
  virtual ~Generator(){}
};


class SineGenerator: public Generator{
public:
  std::vector<float> m_freq;
  std::vector<float> m_mu;
  std::vector<float> m_amp;
  std::vector<float> m_phase;
  float dx;
  int nEle;
  float m_x=0.;

public:
  SineGenerator(): m_freq{1., 2., 3., 4.}, m_mu{2., 3., 4., 5.},
                   m_amp{1., 2., 3., 4.}, m_phase{0., 1., 2., 0.},
                   dx(0.01), nEle(4){
  }
  SineGenerator(std::vector<float> freq, std::vector<float> mu,
                std::vector<float> amp, std::vector<float> phase,
                float dx): dx(dx){
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

  std::vector<float> next() override {
    std::vector<float> out;
    for(int i=0; i!=nEle; i++){
      out.push_back(m_mu[i]+m_amp[i]*std::sin(2*PI*m_x*m_freq[i]));
    }
        m_x += dx;
    return out;
  }
};
