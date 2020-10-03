#ifndef PYENV_H_
#define PYENV_H_

#include <pybind11/pybind11.h>

#include "Env.h"

namespace madigan{

  class PyEnv: public Env{
  public:
    using Env::Env;


  }

}
#endif
