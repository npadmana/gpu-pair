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

  cout << "Sorting by x" << endl;
  sortParticles(p1, 0, 10, 0);
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }

  cout << "Sorting by y" << endl;
  sortParticles(p1, 0, 10, 1);
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }

  cout << "Sorting by z" << endl;
  sortParticles(p1, 0, 10, 2);
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }


  // Test partial sort
  cout << "Partial sort" << endl;
  sortParticles(p1, 3, 8, 0);
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpackParticle(i,x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }


}
