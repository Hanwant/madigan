#ifndef CONFIG_H_
#define CONFIG_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <any>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataTypes.h"

namespace madigan{
  // =============================================================================
  //============================ CONFIG ==========================================
  // =============================================================================

  using std::vector;
  using std::string;

  typedef std::unordered_map<string, std::any> Config;
  // Synth has same parameters for multiple dataSources (I.e sine wave, sawtooth, sine composite)
  // so dataSourceType is also passed
  Config makeConfigFromPyDict(pybind11::dict dict);
  Config makeSynthConfigFromPyDict(pybind11::dict datasource_pydict, string dataSourceType);
  Config makeSineDynamicConfigFromPyDict(pybind11::dict datasource_pydict);
  Config makeOUConfigFromPyDict(pybind11::dict datasource_pydict);
  Config makeSimpleTrendConfigFromPyDict(pybind11::dict datasource_pydict);
  Config makeTrendOUConfigFromPyDict(pybind11::dict datasource_pydict);
  Config makeHDFSourceConfigFromPyDict(pybind11::dict datasource_pydict);


} // namespace madigan

#endif

// template<typename T>
// struct Config{
// };

// template<typename T>
// struct DataSourceConfig
// template<>
// struct DataSourceConfig<Synth>{
//   vector<double> freq;
//   vector<double> mu;
//   vector<double> amp;
//   vector<double> phase;
//   double dX;
// };

// Config<SynthParams>{
//   SynthParams generator
// }
