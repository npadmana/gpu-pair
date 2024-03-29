#ifndef CPU_UTILS_H_
#define CPU_UTILS_H_

// Collection of utility functions
// Nikhil Padmanabhan, Yale, May 2012

#include <vector>
#include <ctime>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

class GSLRandom {
  private:
    gsl_rng *rng;
  public :
    GSLRandom(unsigned long int seed);
    ~GSLRandom();

    // Return a random number 
    double operator()(double a, double b);

    // Fill a vector
    // This is a little complicated, simply to allow passing in arbitrary storage
    template <typename T> 
    void operator()(double a, double b, T &vec) {
    	typedef typename T::value_type type;
    	typedef typename T::iterator iter;
    	for (iter i=vec.begin(); i != vec.end(); ++i)
    		*i = static_cast<type>(gsl_ran_flat(rng, a, b));
    }

};


class CPUclock {
  private :
    clock_t t0;
  public :
    CPUclock();
    void reset();
    double elapsed();
};


#endif /* UTILS_H_ */
