#include "Particles.h"
#include "utils.h"
#include <thrust/tuple.h>

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


Particles::ParticleIterator Particles::begin()  {
  ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), w.begin())));
  return ii;
}

Particles::ParticleIterator Particles::end()  {
  ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), w.end())));
  return ii;
}

void unpackParticle(Particles::ParticleIterator& tup, float& x, float& y, float& z, int& w) {
	auto t1 = *tup;
	x = thrust::get<0>(t1);
	y = thrust::get<1>(t1);
	z = thrust::get<2>(t1);
	w = thrust::get<3>(t1);
}


