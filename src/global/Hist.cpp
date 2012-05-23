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


void diffHist(const RHist &r1, const RHist &r2) {

	auto jj = r2.hist.begin();
	unsigned long long diff=0ull, val=0ull, tmp=0ull, diff1=0ull;
	for_each(r1.hist.begin(), r1.hist.end(), [&](unsigned long long ii) {
		tmp = *jj;
		if (tmp > ii) {diff1 = tmp-ii;}
		else { diff1 = ii-tmp;}
		if (diff1 > diff) {diff = diff1; val = ii;}
	});
	cout << "The largest difference between the two histograms was :: " << diff << endl;
	cout << "The value at that point was :: " << val << endl;
}
