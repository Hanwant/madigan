#ifndef CONFIG_H_
#define CONFIG_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <any>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace madigan{
  // =============================================================================
  //============================ CONFIG ==========================================
  // =============================================================================

  using std::vector;
  using std::string;

  typedef std::unordered_map<string, std::any> Config;
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


  inline Config makeConfigFromPyDict(pybind11::dict dict){
    Config config;
    // Config genParams;
    auto dataSourceFound=std::find_if(dict.begin(), dict.end(),
                                      [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                        return string(pybind11::str(pair.first)) == "data_source_type";
                                      });
    if (dataSourceFound != dict.end()){
      std::string dataSourceType = string(pybind11::str(dataSourceFound->second));
      if ( dataSourceType == "Synth" || dataSourceType == "SawTooth" ||
           dataSourceType == "Triangle" || dataSourceType == "SineAdder"){
          auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                           [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                             return string(pybind11::str(pair.first)) == "generator_params";
                                           });
          if (genParamsFound != dict.end()){
            pybind11::dict genParams = dict[pybind11::str("generator_params")];
            for (auto key: {"freq", "mu", "amp", "phase", "dX", "noise"}){
              auto keyFound=std::find_if(genParams.begin(), dict.end(),
                                         [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == key;
                                         });
              if (keyFound == genParams.end()){
                throw ConfigError(string(key)+" key not found in generator_params in config");
              }
            }
            vector<double> freq;
            vector<double> amp;
            vector<double> mu;
            vector<double> phase;
            double dX;
            double noise;
            for(auto& item: genParams){
              string key = string(pybind11::str(item.first));
              if(key == "freq"){
                freq = item.second.cast<vector<double>>();
              }
              if(key == "mu"){
                mu = item.second.cast<vector<double>>();
              }
              if(key == "amp"){
                amp = item.second.cast<vector<double>>();
              }
              if(key == "phase"){
                phase = item.second.cast<vector<double>>();
              }
              if(key == "dX"){
                dX = item.second.cast<double>();
              }
              if(key == "noise"){
                noise = item.second.cast<double>();
              }
            }
            config=Config({
                {"data_source_type", dataSourceType},
                {"generator_params", Config{{"freq", freq},
                                            {"mu", mu},
                                            {"amp", amp},
                                            {"phase", phase},
                                            {"dX", dX},
                                            {"noise", noise}}
                }
              });
          }
          else{
            throw ConfigError("config for DataSource type Synth needs generator params");
          }
      }
      else if ( dataSourceType == "OU"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "generator_params";
                                           });
          if (genParamsFound != dict.end()){
            pybind11::dict genParams = dict[pybind11::str("generator_params")];
            for (auto key: {"mean", "theta", "phi", "noise_var"}){
              auto keyFound=std::find_if(genParams.begin(), dict.end(),
                                         [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == key;
                                         });
              if (keyFound == genParams.end()){
                throw ConfigError(string(key)+" key not found in generator_params in config");
              }
            }
            vector<double> mean;
            vector<double> theta;
            vector<double> phi;
            vector<double> noise_var;
            for(auto& item: genParams){
              string key = string(pybind11::str(item.first));
              if(key == "mean"){
                mean= item.second.cast<vector<double>>();
              }
              if(key == "theta"){
                theta= item.second.cast<vector<double>>();
              }
              if(key == "phi"){
                phi = item.second.cast<vector<double>>();
              }
              if(key == "noise_var"){
                noise_var = item.second.cast<vector<double>>();
              }
            }
            config=Config({
                {"data_source_type", dataSourceType},
                {"generator_params", Config{{"mean", mean},
                                            {"theta", theta},
                                            {"phi", phi},
                                            {"noise_var", noise_var}}
                }
              });
          }
          else{
            throw ConfigError("config for DataSource type OU needs generator params");
          }
      }
      else{
        std::stringstream ss;
        ss << "(Py->C++) Config Parsing for " << dataSourceType << " has not been implemented";
        throw NotImplemented(ss.str());
      }
    }
    else{
      throw ConfigError("Config needs entry for data_source_type");
    }
    return config;
  }
} // namespace madigan

#endif
