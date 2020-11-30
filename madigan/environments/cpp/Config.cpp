#include "Config.h"

namespace madigan {

   Config makeConfigFromPyDict(pybind11::dict dict){
    Config config;
    // Config genParams;
    auto dataSourceFound=std::find_if(dict.begin(), dict.end(),
                                      [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                        return string(pybind11::str(pair.first)) == "data_source_type";
                                      });
    if (dataSourceFound != dict.end()){
      std::string dataSourceType = string(pybind11::str(dataSourceFound->second));
      // std::cout << dataSourceType << "\n";
      if ( dataSourceType == "Synth" || dataSourceType == "SawTooth" ||
           dataSourceType == "Triangle" || dataSourceType == "SineAdder"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          config = makeSynthConfigFromPyDict(genParams, dataSourceType);
        }
        else{
          throw ConfigError("config for DataSource type Synth needs generator params");
        }
      }
      else if ( dataSourceType == "OU"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          config = makeOUConfigFromPyDict(genParams);
        }
        else{
          throw ConfigError("config for DataSource type OU needs data_source_config");
        }
      }
      else if ( dataSourceType == "SimpleTrend"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          config = makeSimpleTrendConfigFromPyDict(genParams);
        }
        else{
          throw ConfigError("config for DataSource type SimpleTrend needs data_source_config");
        }
      }
      else if(dataSourceType == "Composite"){
        Config data_source_config;
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle,
                                            pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first))
                                             == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          for(auto& genParam: genParams){
            pybind11::dict subDict = genParam.second.cast<pybind11::dict>();
            Config subConfig = makeConfigFromPyDict(subDict);
            string dSourceType = string(pybind11::str(subDict["data_source_type"]));
            data_source_config[dSourceType] = subConfig;
          }
          config["data_source_type"] = "Composite";
          config["data_source_config"] = data_source_config;
        }
      }
      else if ( dataSourceType == "TrendOU"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          config = makeTrendOUConfigFromPyDict(genParams);
        }
        else{
          throw ConfigError("config for DataSource type SimpleTrend needs data_source_config");
        }
      }
      else if ( dataSourceType == "HDFSource"){
        auto genParamsFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (genParamsFound != dict.end()){
          pybind11::dict genParams = dict[pybind11::str("data_source_config")];
          config = makeHDFSourceConfigFromPyDict(genParams);
        }
        else{
          throw ConfigError("config for DataSource type HDFSource needs data_source_config");
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

   Config makeSynthConfigFromPyDict(pybind11::dict genParams, string dataSourceType){
    for (auto key: {"freq", "mu", "amp", "phase", "dX", "noise"}){
      auto keyFound=std::find_if(genParams.begin(), genParams.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == genParams.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
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
    Config config{
      {"data_source_type", dataSourceType},
      {"data_source_config", Config{{"freq", freq},
                                    {"mu", mu},
                                    {"amp", amp},
                                    {"phase", phase},
                                    {"dX", dX},
                                    {"noise", noise}}
      }
    };
    return config;
  }

   Config makeOUConfigFromPyDict(pybind11::dict genParams){
    for (auto key: {"mean", "theta", "phi", "noise_var"}){
      auto keyFound=std::find_if(genParams.begin(), genParams.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == genParams.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
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
    Config config{
      {"data_source_type", "OU"},
      {"data_source_config", Config{{"mean", mean},
                                    {"theta", theta},
                                    {"phi", phi},
                                    {"noise_var", noise_var}}
      }
    };
    return config;
  }

   Config makeSimpleTrendConfigFromPyDict(pybind11::dict genParams){
    for (auto key: {"trend_prob", "min_period", "max_period", "noise", "dYMin",
                    "dYMax", "start"}){
      auto keyFound=std::find_if(genParams.begin(), genParams.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == genParams.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> noise;
    vector<double> dYMin;
    vector<double> dYMax;
    vector<double> start;
    for(auto& item: genParams){
      string key = string(pybind11::str(item.first));
      if(key == "trend_prob"){
        trendProb = item.second.cast<vector<double>>();
      }
      if(key == "min_period"){
        minPeriod= item.second.cast<vector<int>>();
      }
      if(key == "max_period"){
        maxPeriod= item.second.cast<vector<int>>();
      }
      if(key == "noise"){
        noise = item.second.cast<vector<double>>();
      }
      if(key == "dYMin"){
        dYMin = item.second.cast<vector<double>>();
      }
      if(key == "dYMax"){
        dYMax = item.second.cast<vector<double>>();
      }
      if(key == "start"){
        start = item.second.cast<vector<double>>();
      }
    }
    Config config{
      {"data_source_type", "SimpleTrend"},
      {"data_source_config", Config{{"trendProb", trendProb},
                                    {"minPeriod", minPeriod},
                                    {"maxPeriod", maxPeriod},
                                    {"noise", noise },
                                    {"dYMin", dYMin},
                                    {"dYMax", dYMax},
                                    {"start", start}}
      }
    };
    return config;
  }

   Config makeTrendOUConfigFromPyDict(pybind11::dict genParams){
    for (auto key: {"trend_prob", "min_period", "max_period", "dYMin",
                    "dYMax", "start", "theta", "phi", "noise_trend", "ema_alpha"}){
      auto keyFound=std::find_if(genParams.begin(), genParams.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == genParams.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<double> trendProb;
    vector<int> minPeriod;
    vector<int> maxPeriod;
    vector<double> dYMin;
    vector<double> dYMax;
    vector<double> start;
    vector<double> theta;
    vector<double> phi;
    vector<double> noiseTrend;
    vector<double> emaAlpha;
    for(auto& item: genParams){
      string key = string(pybind11::str(item.first));
      if(key == "trend_prob"){
        trendProb = item.second.cast<vector<double>>();
      }
      if(key == "min_period"){
        minPeriod = item.second.cast<vector<int>>();
      }
      if(key == "max_period"){
        maxPeriod= item.second.cast<vector<int>>();
      }
      if(key == "dYMin"){
        dYMin = item.second.cast<vector<double>>();
      }
      if(key == "dYMax"){
        dYMax = item.second.cast<vector<double>>();
      }
      if(key == "start"){
        start = item.second.cast<vector<double>>();
      }
      if(key == "theta"){
        theta = item.second.cast<vector<double>>();
      }
      if(key == "phi"){
        phi = item.second.cast<vector<double>>();
      }
      if(key == "noise_trend"){
        noiseTrend= item.second.cast<vector<double>>();
      }
      if(key == "ema_alpha"){
        emaAlpha = item.second.cast<vector<double>>();
      }
    }
    Config config{
      {"data_source_type", "TrendOU"},
      {"data_source_config", Config{{"trendProb", trendProb},
                                    {"minPeriod", minPeriod},
                                    {"maxPeriod", maxPeriod},
                                    {"dYMin", dYMin},
                                    {"dYMax", dYMax},
                                    {"start", start},
                                    {"theta", theta},
                                    {"phi", phi},
                                    {"noiseTrend", noiseTrend},
                                    {"emaAlpha", emaAlpha}}
      }
    };
    return config;
  }

  Config makeHDFSourceConfigFromPyDict(pybind11::dict genParams){
    for (auto key: {"filepath", "main_key", "price_key", "timestamp_key"}){
      auto keyFound=std::find_if(genParams.begin(), genParams.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == genParams.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    Config config{{"data_source_type", "HDFSource"}};
    for(auto& item: genParams){
      string key = string(pybind11::str(item.first));
      config[key] = item.second.cast<string>();
    }
    return config;
  }
}
