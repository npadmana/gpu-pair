#include <iostream>
#include "Particles.h"
#include "cpu_utils.h"

using namespace std;

int main() {
  CPUParticles p1;
  float x,y,z; int w;

  p1.mkRandom(10, 101);
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }

  auto minmax = minmaxParticle(p1);
  cout << "The minimum is :" ; (thrust::get<0>(minmax)).print(); cout << endl;
  cout << "The maximum is :" ; (thrust::get<1>(minmax)).print(); cout << endl;

}
