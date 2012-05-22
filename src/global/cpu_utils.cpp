#include "cpu_utils.h"

using namespace std;

GSLRandom::GSLRandom(unsigned long int seed) {
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);
}

GSLRandom::~GSLRandom() {
  gsl_rng_free(rng);
}


double GSLRandom::operator()(double a, double b) {
  return gsl_ran_flat(rng, a, b);
}


CPUclock::CPUclock() {
  this->reset();
}

void CPUclock::reset() {
  t0 = clock();
}

double CPUclock::elapsed() {
  clock_t t1;
  t1 = clock();
  return (static_cast<double>(t1-t0))/(static_cast<double>(CLOCKS_PER_SEC)) * 1000.0;
}
