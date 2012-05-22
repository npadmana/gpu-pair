#ifndef CPU_PAIRCOUNTS_H_
#define CPU_PAIRCOUNTS_H_ 

#include "Particles.h"
#include "Hist.h"

namespace CPU_PairCounts
{
  // Naive pair counter 
  void naiveR(Particles &p1, Particles &p2, RHist& rhist);


};



#endif /* CPU_PAIRCOUNTS_H_ */
