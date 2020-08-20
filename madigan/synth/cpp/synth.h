#include <string>

class SineGen{
 private:
  double mu=0.;
  double amp=1.;
  double dx=0.01;
  double x=0.;

 public:
  SineGen(){}
 SineGen(double mu, double amp, double dx): mu(mu), amp(amp), dx(dx){}
  double render();
};

