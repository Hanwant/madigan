#include "synth.h"
#include <iostream>
#include <math.h>


double SineGen::render(){
  double y = mu + amp * std::sin(x);
  x += dx;
  return y;
}

int main(){

  SineGen generator = SineGen();
  for(int i=0; i < 10000; i++){
    // std::cout << mu + amp * std::sin(x) << std::endl;
    // x += dx;
    std::cout << generator.render() << std::endl;
  }



  return 0;
}
