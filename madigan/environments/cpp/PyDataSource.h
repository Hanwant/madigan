#ifndef PYDATA_SOURCE_H_
#define PYDATA_SOURCE_H_

#include "DataSource.h"

namespace madigan{
  class PyDataSource: public DataSourceTick{
  public:
    using DataSourceTick::DataSourceTick;
    const PriceVector& getData() override {
      PYBIND11_OVERLOAD(const PriceVector&,
                        DataSourceTick,
                        getData,);
    }
  };
  class PyHDFSource: public HDFSource{
  public:
    using HDFSource::HDFSource;
    const PriceVector& getData() override {
      PYBIND11_OVERLOAD(const PriceVector&,
                        HDFSource,
                        getData,);
    }
  };
}

#endif
