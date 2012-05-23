#include <iostream>
#include <algorithm>
#include <cstdlib>

#include "CPU_PairCounts.h"

using namespace std;

int main(int argc, char *argv[]) {
	cout << "A simple test of the pair counting code" << endl;

	int Npart = atoi(argv[1]);
	int Nbins = atoi(argv[2]);
	cout << "Using N particles = " << Npart << endl;
	cout << "Using N histogram bins = " << Nbins << endl;

	RHist rr(Nbins, 0.0, 1.0/static_cast<double>(Nbins));

	CPUParticles p1, p2;
	{
		CPUclock t1;
		p1.mkRandom(Npart, 111);
		p2.mkRandom(Npart, 129);
		double dt = t1.elapsed();
		cout << "Time to generate random particles (ms) ::" << dt << endl;
	}

	{
		CPUclock t1;
		CPU_PairCounts::naiveR(p1, p2, rr);
		double dt = t1.elapsed();
		cout << "Time to count the pairs (ms) :: " << dt << endl;
	}

}
