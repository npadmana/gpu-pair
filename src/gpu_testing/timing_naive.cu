#include <iostream>
#include <algorithm>
#include <cstdlib>

#include "Particles.h"
#include "CPU_PairCounts.h"
#include "GPU_PairCounts.h"
#include "gpu_utils.h"

using namespace std;

int main(int argc, char *argv[]) {
	cout << "A simple test of the pair counting code" << endl;

	// Get parameters
	int Npart = atoi(argv[1]);
	int Nbins = atoi(argv[2]);
	cout << "Using N particles = " << Npart << endl;
	cout << "Using N histogram bins = " << Nbins << endl;

	// Set up the histogram
	RHist cpu_rr(Nbins, 0.0, 1.0/static_cast<double>(Nbins));
	RHist gpu_rr(Nbins, 0.0, 1.0/static_cast<double>(Nbins));
	
	// Initialize the particles
	CPUParticles p1, p2;
	{
		CPUclock t1;
		p1.mkRandom(Npart, 111);
		p2.mkRandom(Npart, 129);
		double dt = t1.elapsed();
		cout << "Time to generate random particles (ms) ::" << dt << endl;
	}

	// Pair counting on the CPU
	{
		CPUclock t1;
		CPU_PairCounts::naiveR(p1, p2, cpu_rr);
		double dt = t1.elapsed();
		cout << "Time to count the pairs on the CPU (ms) :: " << dt << endl;
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
	
	
	// Pair counting on the GPU
	{
		cudaDeviceProp prop;
		cuda_safe_call(cudaGetDeviceProperties(&prop, 0));
		int Nblocks = 2*prop.multiProcessorCount;
		int Nthreads = 512;
		GPUclock t1;
		GPU_PairCounts::naiveR(Nblocks, Nthreads, g1, g2, gpu_rr);
		float dt = t1.elapsed();
		cout << "Time to move the particles to the GPU (ms) ::" << dt << endl;
	}
	
	// Compare histograms
	diffHist(cpu_rr, gpu_rr);

}
