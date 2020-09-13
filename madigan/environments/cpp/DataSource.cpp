#include <stdexcept>
#include <iostream>
#include "DataSource.h"

namespace madigan{

  void Synth::initParams(std::vector<double> _freq, std::vector<double> _mu,
                         std::vector<double> _amp, std::vector<double> _phase,
                         double _dX)
  {
    std::cout << "initing params\n";
    this->freq=_freq;
    this->mu=_mu;
    this->amp=_amp;
    this->initPhase=_phase;
    this->x=_phase;
    this->nAssets=_freq.size();
    this->dX=_dX;
    currentData.resize(nAssets);;
  }

  Synth::Synth(){
    double _dX{0.01};
    vector<double> freq{1., 0.3, 2., 0.5};
    vector<double> mu{2., 2.1, 2.2, 2.3};
    vector<double> amp{1., 1.2, 1.3, 1.};
    vector<double> phase{0., 1., 2., 1.};
    initParams(freq, mu, amp, phase, dX);
  }
  Synth::Synth(std::vector<double> _freq, std::vector<double> _mu,
               std::vector<double> _amp, std::vector<double> _phase,
               double _dX){
    if ((_freq.size() == _mu.size()) && (_mu.size() == _amp.size()) &&
        (_amp.size() == _phase.size())){
      initParams(_freq, _mu, _amp, _phase, _dX);
    }
    else{
      throw std::length_error("parameters passed to DataSource of type Synth"
                              " need to be vectors of same length");
    }
  }


  const PriceVector& Synth::getData(){
    for (int i=0; i < nAssets; i++){
      currentData[i] = mu[i] + amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    return currentData;
  }

  const pybind11::array_t<double> Synth::getData_np(){
    for (int i=0; i < nAssets; i++){
      currentData[i] = mu[i] + amp[i] * std::sin(PI2*x[i]*freq[i]);
      x[i] += dX;
    }
    return pybind11::array_t<double>(
                               {currentData.size()},
                               {sizeof(double)},
                               currentData.data()
                               );
  }


}// namespace madigan
