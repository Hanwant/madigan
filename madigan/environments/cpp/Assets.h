#ifndef ASSET_H_
#define ASSET_H_
#include <string>
#include <vector>

using namespace std;

namespace madigan{

  struct Asset{
    string code;
    string exchange;
    string name;
    int bpMultiplier;
    Asset(string assetName): name(assetName), code(assetName){
    };
    Asset(string assetName, string exchange): name(assetName), code(assetName),
                                              exchange(exchange){
    };
  };

  struct Assets :public std::vector<Asset>{
    Assets(){};
    Assets(vector<string> assetNames){
      for(auto name: assetNames){
        push_back(Asset(name));
      }
    }
    Assets(vector<Asset> assets){
      for(auto asset: assets){
        push_back(asset);
      }
    }
  };

}

#endif
