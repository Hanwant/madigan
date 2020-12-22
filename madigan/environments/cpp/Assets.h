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
    bool operator==(const Asset& other){
      if(code == other.code && exchange == other.exchange){
        return true;
      } else return false;
    }
  };

  inline std::ostream& operator<<(std::ostream& os, Asset asset){
    os << asset.code;
    return os;
  }

  struct Assets :public vector<Asset>{
    Assets(){};
    // using vector<Asset>::vector<Asset>(std::initializer_list);
    using vector::vector;
    Assets(std::initializer_list<string> assetNames){
      for(auto name: assetNames){
        push_back(Asset(name));
      }
    };
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

  inline std::ostream& operator<<(std::ostream& os, Assets assets){
    for (auto& asset: assets){
      os << asset << (string)" ";
    }
    return os;
  }

  // typedef vector<Asset> Assets;
  // template<>
  // Assets::Assets(vector<string> assetNames){
  //   for (auto name: assetNames){
  //     push_back(Asset(name));
  //   }
  // }

}

#endif
