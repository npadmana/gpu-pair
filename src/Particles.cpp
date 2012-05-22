#include "Particles.h"
#include "utils.h"

using namespace std;


void Particles::resize(int N) {
  Npart = N;
  x.resize(N); y.resize(N); z.resize(N); w.resize(N);
}


void Particles::mkRandom(int N, unsigned long int seed) {

  // Resize the vectors
  this->resize(N);

  // Fill these up
  GSLRandom rnd(seed);
  rnd(0.0, 1.0, x); 
  rnd(0.0, 1.0, y); 
  rnd(0.0, 1.0, z); 
  fill(w.begin(), w.end(), 1);
}


auto Particles::begin() const -> decltype(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), w.begin()))) {
  return thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), w.begin()));
}

auto Particles::end() const -> decltype(thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), w.end()))) {
  return thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), w.end()));
}

