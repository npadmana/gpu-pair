#include <iostream>
#include "Particles.h"
#include "cpu_utils.h"

using namespace std;

int main() {
  CPUParticles p1, p2;
  float x,y,z; int w;

  p1.mkRandom(10, 101);

  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }

  moveParticles(p1, p2, 13);
  for (auto i=p2.begin(); i !=p2.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }

}
