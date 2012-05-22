#ifndef PARTICLES_H_
#define PARTICLES_H_ 

#include <vector>
#include "thrust/iterator/zip_iterator.h"

class Particles {
  public :
    int Npart;
    std::vector<float> x,y,z;
    std::vector<int> w;
    typedef std::vector<float>::iterator FloatIterator;
    typedef std::vector<int>::iterator IntIterator;
    typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator, IntIterator> TupleIterator;
    typedef thrust::zip_iterator<TupleIterator> ParticleIterator;


    // Functions
    
    // Resize storage to hold N particles
    void resize(int N);

    // Fill the vectors with random numbers between 0 to 1
    // The weights get filled with 1 automatically
    // This is mostly for testing
    void mkRandom(int N, unsigned long int seed);


    // zip_iterators
    ParticleIterator begin();
    ParticleIterator end();
};
    

void unpackParticle(Particles::ParticleIterator &tup, float& x, float& y, float& z, int& w);

#endif /* PARTICLES_H_ */