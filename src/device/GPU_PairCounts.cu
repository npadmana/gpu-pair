#include "GPU_PairCounts.h"
#include <thrust/device_vector.h>

__global__ void naive_r_kernel
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
	
	naive_r_kernel<<<Nblocks, Nthreads>>> (p1.Npart, thrust::raw_pointer_cast(&p1.x[0]), thrust::raw_pointer_cast(&p1.y[0]), 
			thrust::raw_pointer_cast(&p1.z[0]), thrust::raw_pointer_cast(&p1.w[0]),
			p2.Npart, thrust::raw_pointer_cast(&p2.x[0]), thrust::raw_pointer_cast(&p2.y[0]), 
			thrust::raw_pointer_cast(&p2.z[0]), thrust::raw_pointer_cast(&p2.w[0]),
			rr.Nbins, rr.rmin, rr.dr, thrust::raw_pointer_cast(&hist[0]));
	
	
	thrust::copy(hist.begin(), hist.end(), rr.hist.begin());
	
}

__global__ void shared_r_kernel
(int N1, float *x1, float *y1, float *z1, int *w1,
	int N2, float *x2, float *y2, float *z2, int *w2,
	int Nh, float rmin, float dr, unsigned long long *hist) {

	// Keep a shared copy of the histogram
	__shared__ long long _hist[BUFHIST];

	// We distribute p1, but loop through all of p2
	int ii, jj, idr, nh1, ih, hstart, hend;
	int stride = blockDim.x * gridDim.x;
	float x, y, z, dx, dy, dz, rr, _w1, _w2;

	// Compute the number of histograms
	nh1 = (Nh + BUFHIST - 1)/BUFHIST;

	// Do each piece of the histogram separately
	for (ih = 0; ih < nh1; ++ih) {
		// Define histogram piece
		hstart = ih*BUFHIST;
		hend = hstart + BUFHIST;
		if (hend > Nh) hend = Nh;


		// Zero histogram
		ii = threadIdx.x;
		while (ii < BUFHIST) {
			_hist[ii] = 0ll;
			ii += blockDim.x;
		}
		__syncthreads();

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
				if ((idr < hend) && (idr >= hstart)) atomicAdd( (unsigned long long*) &_hist[idr-hstart], _w2*_w1);
			}
			ii += stride;
		}

		// Synchronize
		__syncthreads();

		// Copy histogram 
		ii = threadIdx.x + hstart;
		while (ii < hend) {
			atomicAdd( (unsigned long long*) &hist[ii], _hist[ii-hstart]);
			ii += blockDim.x;
		}
		__syncthreads();
	}
}
	
	
void GPU_PairCounts::sharedR(int Nblocks, int Nthreads, GPUParticles& p1, GPUParticles& p2, RHist& rr) {
	// Copy the histogram onto the device --- we track the previous values
	thrust::device_vector<unsigned long long> hist(rr.hist);
		
	shared_r_kernel<<<Nblocks, Nthreads>>> (p1.Npart, thrust::raw_pointer_cast(&p1.x[0]), thrust::raw_pointer_cast(&p1.y[0]), 
			thrust::raw_pointer_cast(&p1.z[0]), thrust::raw_pointer_cast(&p1.w[0]),
			p2.Npart, thrust::raw_pointer_cast(&p2.x[0]), thrust::raw_pointer_cast(&p2.y[0]), 
			thrust::raw_pointer_cast(&p2.z[0]), thrust::raw_pointer_cast(&p2.w[0]),
			rr.Nbins, rr.rmin, rr.dr, thrust::raw_pointer_cast(&hist[0]));
	
	
	thrust::copy(hist.begin(), hist.end(), rr.hist.begin());
	
}


__global__ void shared_buffered_r_kernel
(int N1, float *x1, float *y1, float *z1, int *w1,
	int N2, float *x2, float *y2, float *z2, int *w2,
	int Nh, float rmin, float dr, unsigned long long *hist) {

	// Keep a shared copy of the histogram
	__shared__ long long _hist[BUFHIST];
	
	// Allocate particle buffers 
	__shared__ float xbuf[PARTICLEBUFFER], ybuf[PARTICLEBUFFER], zbuf[PARTICLEBUFFER];
	__shared__ int wbuf[PARTICLEBUFFER];

	// We distribute p1, but loop through all of p2
	int ii, jj, idr, nh1, ih, hstart, hend, nbuf, kk;
	int stride = blockDim.x * gridDim.x;
	float x, y, z, dx, dy, dz, rr, _w1, _w2;

	// Compute the number of histograms
	nh1 = (Nh + BUFHIST - 1)/BUFHIST;
	
	// Compute the number of buffered runs we need
	nbuf = (N2 + PARTICLEBUFFER - 1)/PARTICLEBUFFER;

	// Do each piece of the histogram separately
	for (ih = 0; ih < nh1; ++ih) {
		// Define histogram piece
		hstart = ih*BUFHIST;
		hend = hstart + BUFHIST;
		if (hend > Nh) hend = Nh;

		// Zero histogram
		// NOTE : we do not need __syncthreads here, since the end of the loop also does a syncthreads
		ii = threadIdx.x;
		while (ii < BUFHIST) {
			_hist[ii] = 0ll;
			ii += blockDim.x;
		}
		__syncthreads();

		
		// Pair counting happens here -- outside loop
		ii = threadIdx.x + blockIdx.x * blockDim.x;
		while (ii < N1) {
			x = x1[ii]; y = y1[ii]; z = z1[ii]; _w1 = w1[ii];
			
			for (kk = 0; kk < nbuf; ++kk) {
				
				// Fill the particle buffer -- synchronize on both sides	
				__syncthreads();
				idr = kk*PARTICLEBUFFER; // Offset in the main particle array -- reuse idr here
 				jj = threadIdx.x;
				while (jj < PARTICLEBUFFER) {
					// This might trigger a little branch divergence at the end, but that's fine
					if ((jj+idr) < N2) {
						xbuf[jj] = x2[jj+idr]; 
						ybuf[jj] = y2[jj+idr];
						zbuf[jj] = z2[jj+idr];
						wbuf[jj] = w2[jj+idr];
					} else {
						xbuf[jj] = 0.0; ybuf[jj] = 0.0; zbuf[jj] = 0.0; wbuf[jj] = 0;
					}
					jj += blockDim.x;
				}
				__syncthreads();
				
				// Now do all the work
				for (jj = 0; jj < PARTICLEBUFFER; ++jj) {
					dx = xbuf[jj] - x;
					dy = ybuf[jj] - y;
					dz = zbuf[jj] - z;
					_w2 = wbuf[jj];
					rr = sqrtf(dx*dx + dy*dy + dz*dz);
					idr = (int) ((rr-rmin)/dr);
					if ((idr < hend) && (idr >= hstart)) atomicAdd( (unsigned long long*) &_hist[idr-hstart], _w2*_w1);
				}
				// No need to synchronize here, since that happends up top
				
			} // End of 2nd particle loop
			
			
			ii += stride;
		} // End of first particle loop

		// Synchronize histogram 
		__syncthreads();
		ii = threadIdx.x + hstart;
		while (ii < hend) {
			atomicAdd( (unsigned long long*) &hist[ii], _hist[ii-hstart]);
			ii += blockDim.x;
		}
		__syncthreads();
	}
}


void GPU_PairCounts::sharedbufferedR(int Nblocks, int Nthreads, GPUParticles& p1, GPUParticles& p2, RHist& rr) {
	// Copy the histogram onto the device --- we track the previous values
	thrust::device_vector<unsigned long long> hist(rr.hist);
	
	// Assert if numbers of particles are not commensurate with number of threads
	if ((p1.Npart%Nthreads)!=0) throw "p1 not commensurate with Nthreads";
	if ((p2.Npart%Nthreads)!=0) throw "p2 not commensurate with Nthreads";
	
	shared_buffered_r_kernel<<<Nblocks, Nthreads>>> (p1.Npart, thrust::raw_pointer_cast(&p1.x[0]), thrust::raw_pointer_cast(&p1.y[0]), 
			thrust::raw_pointer_cast(&p1.z[0]), thrust::raw_pointer_cast(&p1.w[0]),
			p2.Npart, thrust::raw_pointer_cast(&p2.x[0]), thrust::raw_pointer_cast(&p2.y[0]), 
			thrust::raw_pointer_cast(&p2.z[0]), thrust::raw_pointer_cast(&p2.w[0]),
			rr.Nbins, rr.rmin, rr.dr, thrust::raw_pointer_cast(&hist[0]));
	
	
	thrust::copy(hist.begin(), hist.end(), rr.hist.begin());
	
}


