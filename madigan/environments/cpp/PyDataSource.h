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
  class PyHDFSourceSingle: public HDFSourceSingle{
  public:
    using HDFSourceSingle::HDFSourceSingle;
    const PriceVector& getData() override {
      PYBIND11_OVERLOAD(const PriceVector&,
                        HDFSourceSingle,
                        getData,);
    }
  };
}

#endif
