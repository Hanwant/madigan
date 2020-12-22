#include "Config.h"

namespace madigan {

   Config makeConfigFromPyDict(pybind11::dict dict){
    Config config;
    // Config datasource_pydict;
    auto dataSourceFound=std::find_if(dict.begin(), dict.end(),
                                      [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                        return string(pybind11::str(pair.first)) == "data_source_type";
                                      });
    if (dataSourceFound != dict.end()){
      std::string dataSourceType = string(pybind11::str(dataSourceFound->second));
      // std::cout << dataSourceType << "\n";
      if ( dataSourceType == "Synth" || dataSourceType == "SawTooth" ||
           dataSourceType == "Triangle" || dataSourceType == "SineAdder"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeSynthConfigFromPyDict(datasource_pydict, dataSourceType);
        }
        else{
          throw ConfigError("config for DataSource type Synth needs generator params");
        }
      }
      else if ( dataSourceType == "SineDynamic"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                                 [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                                   return string(pybind11::str(pair.first)) == "data_source_config";
                                                 });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeSineDynamicConfigFromPyDict(datasource_pydict);
        }
        else{
          throw ConfigError("config for DataSource type SineDynamic needs data_source_config");
        }
      }
      else if ( dataSourceType == "SineDynamicTrend"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                                 [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                                   return string(pybind11::str(pair.first)) == "data_source_config";
                                                 });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeSineDynamicTrendConfigFromPyDict(datasource_pydict);
        }
        else{
          throw ConfigError("config for DataSource type SineDynamicTrend needs data_source_config");
        }
      }
      else if ( dataSourceType == "OU"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeOUConfigFromPyDict(datasource_pydict);
        }
        else{
          throw ConfigError("config for DataSource type OU needs data_source_config");
        }
      }
      else if ( dataSourceType == "SimpleTrend"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeSimpleTrendConfigFromPyDict(datasource_pydict);
        }
        else{
          throw ConfigError("config for DataSource type SimpleTrend needs data_source_config");
        }
      }
      else if(dataSourceType == "Composite"){
        Config data_source_config;
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle,
                                            pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first))
                                             == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          for(auto& genParam: datasource_pydict){
            pybind11::dict subDict = genParam.second.cast<pybind11::dict>();
            Config subConfig = makeConfigFromPyDict(subDict);
            string dSourceType = string(pybind11::str(subDict["data_source_type"]));
            data_source_config[dSourceType] = subConfig;
          }
          config["data_source_type"] = "Composite";
          config["data_source_config"] = data_source_config;
        }
      }
      else if ( dataSourceType == "TrendOU" ||
                dataSourceType == "TrendyOU"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeTrendOUConfigFromPyDict(datasource_pydict, dataSourceType);
        }
        else{
          throw ConfigError("config for DataSource type TrendOU/TrendyOU needs data_source_config");
        }
      }
      else if ( dataSourceType == "HDFSource"){
        auto datasource_pydictFound=std::find_if(dict.begin(), dict.end(),
                                         [](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                           return string(pybind11::str(pair.first)) == "data_source_config";
                                         });
        if (datasource_pydictFound != dict.end()){
          pybind11::dict datasource_pydict = dict[pybind11::str("data_source_config")];
          config = makeHDFSourceConfigFromPyDict(datasource_pydict);
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

   Config makeSynthConfigFromPyDict(pybind11::dict datasource_pydict, string dataSourceType){
    for (auto key: {"freq", "mu", "amp", "phase", "dX", "noise"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<double> freq;
    vector<double> amp;
    vector<double> mu;
    vector<double> phase;
    double dX;
    double noise;
    for(auto& item: datasource_pydict){
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

  Config makeSineDynamicConfigFromPyDict(pybind11::dict datasource_pydict){
    for (auto key: {"freqRange", "muRange", "ampRange", "dX", "noise"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<std::array<double, 3>> freqRange;
    vector<std::array<double, 3>> ampRange;
    vector<std::array<double, 3>> muRange;
    double dX;
    double noise;
    for(auto& item: datasource_pydict){
      string key = string(pybind11::str(item.first));
      if(key == "freqRange"){
        freqRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "muRange"){
        muRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "ampRange"){
        ampRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "dX"){
        dX = item.second.cast<double>();
      }
      if(key == "noise"){
        noise = item.second.cast<double>();
      }
    }
    Config config{
      {"data_source_type", "SineDynamic"},
      {"data_source_config", Config{{"freqRange", freqRange},
                                    {"muRange", muRange},
                                    {"ampRange", ampRange},
                                    {"dX", dX},
                                    {"noise", noise}}
      }
    };
    return config;
  }

  Config makeSineDynamicTrendConfigFromPyDict(pybind11::dict datasource_pydict){
    for (auto key: {"freqRange", "muRange", "ampRange", "trendRange",
                    "trendProb", "trendIncr", "dX", "noise"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<std::array<double, 3>> freqRange;
    vector<std::array<double, 3>> ampRange;
    vector<std::array<double, 3>> muRange;
    vector<std::array<int, 2>> trendRange;
    vector<double> trendIncr;
    vector<double> trendProb;
    double dX;
    double noise;
    for(auto& item: datasource_pydict){
      string key = string(pybind11::str(item.first));
      if(key == "freqRange"){
        freqRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "muRange"){
        muRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "ampRange"){
        ampRange = item.second.cast<vector<std::array<double, 3>>>();
      }
      if(key == "trendRange"){
        trendRange = item.second.cast<vector<std::array<int, 2>>>();
      }
      if(key == "trendIncr"){
        trendIncr = item.second.cast<vector<double>>();
      }
      if(key == "trendProb"){
        trendProb = item.second.cast<vector<double>>();
      }
      if(key == "dX"){
        dX = item.second.cast<double>();
      }
      if(key == "noise"){
        noise = item.second.cast<double>();
      }
    }
    Config config{
      {"data_source_type", "SineDynamicTrend"},
      {"data_source_config", Config{{"freqRange", freqRange},
                                    {"muRange", muRange},
                                    {"ampRange", ampRange},
                                    {"trendRange", trendRange},
                                    {"trendIncr", trendIncr},
                                    {"trendProb", trendProb},
                                    {"dX", dX},
                                    {"noise", noise}}
      }
    };
    return config;
  }


   Config makeOUConfigFromPyDict(pybind11::dict datasource_pydict){
    for (auto key: {"mean", "theta", "phi", "noise_var"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    vector<double> mean;
    vector<double> theta;
    vector<double> phi;
    vector<double> noise_var;
    for(auto& item: datasource_pydict){
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

   Config makeSimpleTrendConfigFromPyDict(pybind11::dict datasource_pydict){
    for (auto key: {"trend_prob", "min_period", "max_period", "noise", "dYMin",
                    "dYMax", "start"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
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
    for(auto& item: datasource_pydict){
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

  Config makeTrendOUConfigFromPyDict(pybind11::dict datasource_pydict,
                                     string dataSourceType){ // type can be TrendOU or TrendyOU
    for (auto key: {"trend_prob", "min_period", "max_period", "dYMin",
                    "dYMax", "start", "theta", "phi", "noise_trend", "ema_alpha"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
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
    for(auto& item: datasource_pydict){
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
      {"data_source_type", dataSourceType},
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

  Config makeHDFSourceConfigFromPyDict(pybind11::dict datasource_pydict){
    for (auto key: {"filepath", "main_key", "price_key", "timestamp_key"}){
      auto keyFound=std::find_if(datasource_pydict.begin(), datasource_pydict.end(),
                                 [key](const std::pair<pybind11::handle, pybind11::handle>& pair){
                                   return string(pybind11::str(pair.first)) == key;
                                 });
      if (keyFound == datasource_pydict.end()){
        throw ConfigError(string(key)+" key not found in data_source_config in config");
      }
    }
    Config config{{"data_source_type", "HDFSource"}};
    Config data_source_config;
    for(auto& item: datasource_pydict){
      string key = string(pybind11::str(item.first));
      data_source_config[key] = item.second.cast<string>();
    }
    config["data_source_config"] = data_source_config;
    return config;
  }
}
