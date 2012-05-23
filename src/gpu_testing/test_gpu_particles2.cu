#include <iostream>
#include <algorithm>

#include "Particles.h"
#include "CPU_PairCounts.h"
#include "GPU_PairCounts.h"
#include "gpu_utils.h"

using namespace std;

int main() {
	cout << "A simple test of the pair counting code" << endl;
	RHist cpu_rr(20, 0.0, 0.1);
	RHist gpu_rr(20, 0.0, 0.1);

	CPUParticles p1, p2;
	GPUParticles g1, g2;
	
	p1.mkRandom(1000, 111);
	p2.mkRandom(2000, 129);

	moveParticles(p1, g1);
	moveParticles(p2, g2);
	
	CPU_PairCounts::naiveR(p1, p2, cpu_rr);
	
	cudaDeviceProp prop;
	cuda_safe_call(cudaGetDeviceProperties(&prop, 0));
	int Nblocks = 2*prop.multiProcessorCount;
	int Nthreads = 512;
	
	GPU_PairCounts::naiveR(Nblocks,Nthreads,g1, g2, gpu_rr);
	
	cout << "The CPU histogram is :";
	cpu_rr.print();
	
	cout << "The GPU histogram is :";
	gpu_rr.print();

}
