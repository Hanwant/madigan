#include <stdexcept>
#include <iostream>
#include "DataSource.h"

namespace madigan{

  void Synth::initParams(std::vector<double> freq, std::vector<double> mu,
                         std::vector<double> amp, std::vector<double> phase,
                         double dX)
  {
    this->freq=freq;
    this->mu=mu;
    this->amp=amp;
    this->initPhase=phase;
    this->x=phase;
    this->nAssets=freq.size();
    this->dX=dX;
  }

  Synth::Synth(){
    double dX{0.01};
    vector<double> freq{1., 0.3, 2., 0.5};
    vector<double> mu{2., 2.1, 2.2, 2.3};
    vector<double> amp{1., 1.2, 1.3, 1.};
    vector<double> phase{0., 1., 2., 1.};
    initParams(freq, mu, amp, phase, dX);
  }
  Synth::Synth(std::vector<double> freq, std::vector<double> mu,
               std::vector<double> amp, std::vector<double> phase,
               double dX){
    if ((freq.size() == mu.size()) && (mu.size() == amp.size()) &&
        (amp.size() == phase.size())){
      initParams(freq, mu, amp, phase, dX);
    }
    else{
      throw std::length_error("parameters passed to DataSource of type Synth"
                              " need to be vectors of same length");
    }
  }


  std::vector<double> Synth::getData(){
    std::vector<double> out(nAssets);
    for (int i=0; i < nAssets; i++){
      out[i] = mu[i] + amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    return out;
  }


}// namespace madigan
