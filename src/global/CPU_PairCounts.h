#ifndef CPU_PAIRCOUNTS_H_
#define CPU_PAIRCOUNTS_H_ 

#include "Particles.h"
#include "Hist.h"

namespace CPU_PairCounts
{

  // Naive pair counter 
  void naiveR(CPUParticles &p1, CPUParticles &p2, RHist& rhist);


};



#endif /* CPU_PAIRCOUNTS_H_ */
