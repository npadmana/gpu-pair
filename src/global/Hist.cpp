#include "Hist.h"
#include <algorithm>

RHist::RHist(int n, float r0, float _dr) {
  Nbins = n;
  rmin = r0;
  dr = _dr;

  hist.resize(Nbins);
  std::fill(hist.begin(), hist.end(), 0ll);
}
