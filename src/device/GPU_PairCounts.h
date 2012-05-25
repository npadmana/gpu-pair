/*
 * GPU_PairCounts.h
 *
 *  Created on: May 23, 2012
 *      Author: npadmana
 */

#ifndef GPU_PAIRCOUNTS_H_
#define GPU_PAIRCOUNTS_H_

#include "Particles.h"
#include "Hist.h"

const int BUFHIST=1024;
const int PARTICLEBUFFER=1024;

namespace GPU_PairCounts {

	void naiveR(int Nblocks, int Nthreads, GPUParticles &p1, GPUParticles &p2, RHist& rhist);
	void sharedR(int Nblocks, int Nthreads, GPUParticles& p1, GPUParticles& p2, RHist& rr);
	void sharedbufferedR(int Nblocks, int Nthreads, GPUParticles& p1, GPUParticles& p2, RHist& rr);
}



#endif /* GPU_PAIRCOUNTS_H_ */
