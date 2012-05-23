#ifndef PARTICLES_H_
#define PARTICLES_H_ 

#include <vector>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "cpu_utils.h"

template <class FloatVec, class IntVec>
class Particles {
  public :
	// Typedefs
	typedef typename FloatVec::iterator FloatIterator;
	typedef typename IntVec::iterator IntIterator;
    typedef thrust::tuple< FloatIterator, FloatIterator, FloatIterator, IntIterator > TupleIterator;
    typedef thrust::zip_iterator<TupleIterator> ParticleIterator;
	typedef typename FloatVec::const_iterator const_FloatIterator;
	typedef typename IntVec::const_iterator const_IntIterator;
    typedef thrust::tuple< const_FloatIterator, const_FloatIterator, const_FloatIterator, const_IntIterator > const_TupleIterator;
    typedef thrust::zip_iterator<const_TupleIterator> const_ParticleIterator;



	// Members
    int Npart;
    FloatVec x,y,z;
    IntVec w;

    // Functions
    
    // Resize storage to hold N particles
    void resize(int N) {
    	  Npart = N;
    	  x.resize(N); y.resize(N); z.resize(N); w.resize(N);
    }

    // Fill the vectors with random numbers between 0 to 1
    // The weights get filled with 1 automatically
    // This is mostly for testing
    void mkRandom(int N, unsigned long int seed) {
    	  // Resize the vectors
    	  this->resize(N);

    	  // Fill these up
    	  GSLRandom rnd(seed);
    	  rnd(0.0, 1.0, x);
    	  rnd(0.0, 1.0, y);
    	  rnd(0.0, 1.0, z);
    	  thrust::fill(w.begin(), w.end(), 1);
    }

    // zip_iterators
    ParticleIterator begin() {
    	ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), w.begin())));
    	return ii;
    }

    ParticleIterator end() {
    	ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), w.end())));
    	return ii;
    }

    const_ParticleIterator cbegin() const {
    	const_ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.cbegin(), y.cbegin(), z.cbegin(), w.cbegin())));
    	return ii;
    }

    const_ParticleIterator cend() const {
    	const_ParticleIterator ii(thrust::make_zip_iterator(thrust::make_tuple(x.cend(), y.cend(), z.cend(), w.cend())));
    	return ii;
    }

};

template <typename F1, typename I1, typename F2, typename I2>
void moveParticles(const Particles<F1, I1>& p1, Particles<F2, I2>& p2) {
	// Make sure there is enough space
	p2.resize(p1.Npart);

	// Move
	thrust::copy(p1.cbegin(), p1.cend(), p2.begin());
}

template <typename T>
void unpackParticle(const T &tup, float& x, float& y, float& z, int& w) {
	thrust::tuple<float, float, float, int> t1 = *tup; // Normally, this should have just been auto, but work-around for Eclipse
	x = thrust::get<0>(t1);
	y = thrust::get<1>(t1);
	z = thrust::get<2>(t1);
	w = thrust::get<3>(t1);
}




// Particles typedefs
typedef Particles<thrust::host_vector<float>, thrust::host_vector<int> > CPUParticles;
#ifdef __CUDACC__
typedef Particles<thrust::device_vector<float>, thrust::device_vector<int> > GPUParticles;
#endif

#endif /* PARTICLES_H_ */
