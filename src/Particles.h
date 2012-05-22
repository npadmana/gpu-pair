#ifndef PARTICLES_H_
#define PARTICLES_H_ 

#include <vector>
#include <thrust/iterator/zip_iterator.h>

class Particles {
  public :
    int Npart;
    std::vector<float> x,y,z;
    std::vector<int> w;


    // Functions
    
    // Resize storage to hold N particles
    void resize(int N);

    // Fill the vectors with random numbers between 0 to 1
    // The weights get filled with 1 automatically
    // This is mostly for testing
    void mkRandom(int N, unsigned long int seed);


    // zip_iterators
    auto begin() const -> decltype(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), w.begin())));
    auto end() const -> decltype(thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), w.end())));
};
    

#endif /* PARTICLES_H_ */
