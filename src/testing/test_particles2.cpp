#include <iostream>
#include <algorithm>

#include "CPU_PairCounts.h"

using namespace std;

int main() {
	cout << "A simple test of the pair counting code" << endl;
	RHist rr(20, 0.0, 0.1);

	CPUParticles p1, p2;
	p1.mkRandom(10, 111);
	p2.mkRandom(20, 129);

	CPU_PairCounts::naiveR(p1, p2, rr);

	unsigned long long sum=0ll;
	cout << "The histogram is :";
	for_each(rr.hist.begin(), rr.hist.end(), [&sum](unsigned long long ii){
		cout << ii << " ";
		sum += ii;});
	cout << endl << "The sum of the histogram is :" << sum << endl;


}
