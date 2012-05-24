#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <numeric>

#include "Particles.h"
#include "CPU_PairCounts.h"
#include "GPU_PairCounts.h"
#include "gpu_utils.h"


using namespace std;

int main(int argc, char *argv[]) {
	cout << "A simple test of the pair counting code" << endl;

	// Do a simple check on the number of arguments
	if (argc != 5) {
		cout << "timing_shared_gpuonly Npart Nbins blockfac Nthreads niter\n";
		exit(1);
	}
	
	// Get parameters
	int Npart = atoi(argv[1]);
	int Nbins = atoi(argv[2]);
	int blockfac = atoi(argv[3]);
	int Nthreads = atoi(argv[4]);
	int niter = atoi(argv[5]);
	cout << "Using N particles = " << Npart << endl;
	cout << "Using N histogram bins = " << Nbins << endl;
	// Determine CUDA properties
	cudaDeviceProp prop;
	cuda_safe_call(cudaGetDeviceProperties(&prop, 0));
	int Nblocks = blockfac*prop.multiProcessorCount;
	cout << "Using Nblocks :" << Nblocks << endl;
	cout << "Using Nthreads :" << Nthreads << endl;


	
	// Initialize the particles
	CPUParticles p1, p2;
	{
		CPUclock t1;
		p1.mkRandom(Npart, 111);
		p2.mkRandom(Npart, 129);
		double dt = t1.elapsed();
		cout << "Time to generate random particles (ms) ::" << dt << endl;
	}
	
	// Move particles to the GPU
	GPUParticles g1, g2;
	{
		GPUclock t1;
		moveParticles(p1, g1);
		moveParticles(p2, g2);
		float dt = t1.elapsed();
		cout << "Time to move the particles to the GPU (ms) ::" << dt << endl;
	}
	

	// Cache the timing information
	vector<float> timing(niter);
	
	for (int ii=0; ii < niter; ++ii)
	// Pair counting on the GPU
	{
		// Set up the histogram
		RHist gpu_rr(Nbins, 0.0, 1.0/static_cast<double>(Nbins));
		
		GPUclock t1;
		GPU_PairCounts::sharedR(Nblocks, Nthreads, g1, g2, gpu_rr);
		timing[ii] = t1.elapsed();
	}
	
	// A little wasteful
	cout << "Minimum time :" << *min_element(timing.begin(), timing.end()) << endl;
	cout << "Maximum time :" << *max_element(timing.begin(), timing.end()) << endl;
	cout << "Average time :" << accumulate(timing.begin(), timing.end(), 0.0)/static_cast<float>(niter);
	
}
