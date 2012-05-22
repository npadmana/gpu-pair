#include "CPU_PairCounts.h"
#include "utils.h"
#include <cmath>

void CPU_PairCounts::naiveR(Particles& p1, Particles& p2, RHist& hh) {

  // Do the paircounts in the most naive manner
  float x1,y1,z1,x2,y2,z2,dx,dy,dz,rr;
  int w1, w2, ibin;

  // Loop over 1
  for (auto ii = p1.begin(); ii != p1.end(); ++ii) {
    unpackParticle(ii, x1, y1, z1, w1);

    // Loop over 2
    for (auto jj = p2.begin(); jj != p2.end(); ++jj) {
      unpackParticle(jj, x2, y2, z2, w2);

      dx = x1-x2; dy=y1-y2; dz = z1-z2;
      rr = sqrt(dx*dx + dy*dy + dz*dz);
      
      ibin = static_cast<int> ((rr - hh.rmin)/hh.dr);
      if ((ibin >= 0) && (ibin < hh.Nbins)) hh.hist[ibin] += w1*w2; 
    }
  }

}
