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

  struct Assets :public std::vector<Asset>{
    Assets(){};
    // using std::vector<Asset>::std::vector<Asset>(std::initializer_list);
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

  // typedef std::vector<Asset> Assets;
  // template<>
  // Assets::Assets(vector<string> assetNames){
  //   for (auto name: assetNames){
  //     push_back(Asset(name));
  //   }
  // }

}

#endif
