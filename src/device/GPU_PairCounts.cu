#include "GPU_PairCounts.h"
#include <thrust/device_vector.h>

__global__ void naive_kernel
	   (int N1, float *x1, float *y1, float *z1, int *w1,
		int N2, float *x2, float *y2, float *z2, int *w2,
		int Nh, float rmin, float dr, unsigned long long *hist) {
	// We distribute p1, but loop through all of p2
	int ii, jj, idr;
	int stride = blockDim.x * gridDim.x;
	float x, y, z, dx, dy, dz, rr;
	int _w1, _w2;
	ii = threadIdx.x + blockIdx.x * blockDim.x;

	while (ii < N1) {
		x = x1[ii]; y = y1[ii]; z = z1[ii]; _w1 = w1[ii]; 
		for (jj = 0; jj < N2; ++jj) {
			dx = x2[jj] - x;
			dy = y2[jj] - y;
			dz = z2[jj] - z;
			_w2 = w2[jj];
			rr = sqrtf(dx*dx + dy*dy + dz*dz);
			idr = (int) ((rr-rmin)/dr);
			if ((idr >=0) && (idr < Nh)) atomicAdd( (unsigned long long*) &hist[idr], _w2*_w1);
		}
		ii += stride;
	}

}

void GPU_PairCounts::naiveR(int Nblocks, int Nthreads, GPUParticles& p1, GPUParticles& p2, RHist& rr) {
	// Copy the histogram onto the device --- we track the previous values
	thrust::device_vector<unsigned long long> hist(rr.hist);
	
	naive_kernel<<<Nblocks, Nthreads>>> (p1.Npart, thrust::raw_pointer_cast(&p1.x[0]), thrust::raw_pointer_cast(&p1.y[0]), 
			thrust::raw_pointer_cast(&p1.z[0]), thrust::raw_pointer_cast(&p1.w[0]),
			p2.Npart, thrust::raw_pointer_cast(&p2.x[0]), thrust::raw_pointer_cast(&p2.y[0]), 
			thrust::raw_pointer_cast(&p2.z[0]), thrust::raw_pointer_cast(&p2.w[0]),
			rr.Nbins, rr.rmin, rr.dr, thrust::raw_pointer_cast(&hist[0]));
	
	
	thrust::copy(hist.begin(), hist.end(), rr.hist.begin());
	
}