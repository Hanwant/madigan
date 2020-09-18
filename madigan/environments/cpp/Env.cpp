#include "Env.h"

namespace madigan{


  Env::Env(DataSource* dataSource, Broker* broker): dataSource_(dataSource), broker_(broker){
    for (auto acc: broker_->accountBook()){
      acc.second.addDataSource(dataSource_);
    }
  };


}
