#ifndef PYENV_H_
#define PYENV_H_

#include <pybind11/pybind11.h>

#include "Env.h"

namespace madigan{

  class PyEnv: public Env{
  public:
    using Env::Env;

    State reset() override {
      PYBIND11_OVERLOAD(State,
                        Env,
                        reset, );
    }
    SRDISingle step() override {
      PYBIND11_OVERLOAD(SRDISingle,
                        Env,
                        step, );
    }
    SRDISingle step(int assetIdx, double units) override {
      PYBIND11_OVERLOAD(SRDISingle,
                        Env,
                        step,
                        assetIdx,
                        units);
    }
    SRDIMulti step(const AmountVector& units) override {
      PYBIND11_OVERLOAD(SRDIMulti,
                        Env,
                        step,
                        units);
    }



  };

}
#endif
