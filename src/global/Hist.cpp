#include "Hist.h"
#include <algorithm>
#include <iostream>

using namespace std;

RHist::RHist(int n, float r0, float _dr) {
  Nbins = n;
  rmin = r0;
  dr = _dr;

  hist.resize(Nbins);
  fill(hist.begin(), hist.end(), 0ll);
}


void RHist::print() {
	for_each(hist.begin(), hist.end(), [](unsigned long long ii){
		cout << ii << " ";
	});
	cout << endl;
}
