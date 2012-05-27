#ifndef PARTICLES_H_
#define PARTICLES_H_ 

#include <vector>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include "NPVector.h"
#include "cpu_utils.h"

template <class FloatVec, class IntVec>
class Particles {
  public :
	// Typedefs
	typedef typename FloatVec::iterator FloatIterator;
	typedef typename IntVec::iterator IntIterator;
    typedef thrust::tuple< FloatIterator, FloatIterator, FloatIterator, IntIterator > TupleIterator;
    typedef thrust::zip_iterator<TupleIterator> ParticleIterator;
    typedef thrust::pair<FloatIterator, FloatIterator> const_mmIter;


	typedef typename FloatVec::const_iterator const_FloatIterator;
	typedef typename IntVec::const_iterator const_IntIterator;
    typedef thrust::tuple< const_FloatIterator, const_FloatIterator, const_FloatIterator, const_IntIterator > const_TupleIterator;
    typedef thrust::zip_iterator<const_TupleIterator> const_ParticleIterator;
    typedef thrust::pair<const_FloatIterator, const_FloatIterator> mmIter;



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
void moveParticles(const Particles<F1, I1>& p1, Particles<F2, I2>& p2, int buffer=1) {
	// Figure out the correct size and pad
	int n2 = ((p1.Npart + buffer - 1)/buffer) * buffer; // Note that the term in parentheses is integer multiplication

	// Make sure there is enough space
	p2.resize(n2);

	// Move
	thrust::copy(p1.cbegin(), p1.cend(), p2.begin());

	// Fill the buffered region
	if ((p1.Npart % buffer) != 0) {
		thrust::fill(p2.begin()+p1.Npart, p2.end(), thrust::make_tuple(p1.x[0], p1.y[0], p1.z[0], 0));
	}
}

template <typename T>
void unpackParticle(const T &tup, float& x, float& y, float& z, int& w) {
	thrust::tuple<float, float, float, int> t1 = *tup; // Normally, this should have just been auto, but work-around for Eclipse
	x = thrust::get<0>(t1);
	y = thrust::get<1>(t1);
	z = thrust::get<2>(t1);
	w = thrust::get<3>(t1);
}

template <typename F1, typename I1>
thrust::pair<NPVector3f, NPVector3f> minmaxParticle(Particles<F1, I1>& p) {

	NPVector3f min, max;
	typedef typename Particles<F1,I1>::mmIter mmtype;

	// X
	{
		mmtype tup = thrust::minmax_element(p.x.begin(), p.x.end());
		min[0] = *(thrust::get<0>(tup));
		max[0] = *(thrust::get<1>(tup));
	}

	// Y
	{
		mmtype tup = thrust::minmax_element(p.y.begin(), p.y.end());
		min[1] = *(thrust::get<0>(tup));
		max[1] = *(thrust::get<1>(tup));
	}

	// Z
	{
		mmtype tup = thrust::minmax_element(p.z.begin(), p.z.end());
		min[2] = *(thrust::get<0>(tup));
		max[2] = *(thrust::get<1>(tup));
	}


	return thrust::make_pair(min, max);

}


template <typename F1, typename I1>
void sortParticles(Particles<F1,I1>& p, int start, int end, int dim) {

	// Define the key
	F1 key;
	key.resize(end-start);

	switch (dim) {
	case 0 :
		thrust::copy(p.x.begin()+start, p.x.begin()+end, key.begin());
		break;
	case 1 :
		thrust::copy(p.y.begin()+start, p.y.begin()+end, key.begin());
		break;
	case 2 :
		thrust::copy(p.z.begin()+start, p.z.begin()+end, key.begin());
		break;

	}

	thrust::sort_by_key(key.begin(), key.end(), p.begin()+start);

}




// Particles typedefs
typedef Particles<thrust::host_vector<float>, thrust::host_vector<int> > CPUParticles;
#ifdef __CUDACC__
typedef Particles<thrust::device_vector<float>, thrust::device_vector<int> > GPUParticles;
#endif

#endif /* PARTICLES_H_ */
