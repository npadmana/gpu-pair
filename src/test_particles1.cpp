#include <iostream>
#include "Particles.h"
#include "utils.h"

using namespace std;

int main() {
  Particles p1;   
  float x,y,z; int w;

  p1.mkRandom(25, 101);
  
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
      unpack4<float, float, float, int>(*i, x,y,z,w);
      cout << x << " " << y << " " << z << " " << w << endl;
  }
}
