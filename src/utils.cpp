#include "utils.h"

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

