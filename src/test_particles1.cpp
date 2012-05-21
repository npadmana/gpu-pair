#include <iostream>
#include "Particles.h"

using namespace std;

int main() {
  Particles p1;   

  p1.mkRandom(25, 101);
  
  for (auto i=p1.begin(); i !=p1.end(); ++i) {
     cout << thrust::get<0>(*i) << " " 
       << thrust::get<1>(*i) << " "
       << thrust::get<2>(*i) << " "
       << thrust::get<3>(*i) << endl;
  }
}
