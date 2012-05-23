#include <iostream>

#include "gpu_utils.h"
#include "cpu_utils.h"
#include "Particles.h"


using namespace std;

int main() {
	CPUParticles p1;

	{
		CPUclock t1;
		p1.mkRandom(1000000, 101);
		double dt = t1.elapsed();
		cout << "Generating a million randoms on CPU (ms) : " << dt << endl;
	}

	{
		GPUclock t2;
		GPUParticles p2;
		moveParticles(p1, p2);
		float dt = t2.elapsed();
		cout << "Moving data to device (ms) :" << dt << endl;
	}
}
