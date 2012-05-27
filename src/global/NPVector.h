/*
 * myvec.h
 *
 *  Created on: May 27, 2012
 *      Author: npadmana
 */

#ifndef NPVECTOR_H_
#define NPVECTOR_H_

#include <vector>
#include <iostream>

// Simple vector operations
// Define this to get around the major Eigen/NVCC issues

template <typename T, int N>
class NPVector {
	typedef typename std::vector<T>::iterator iter;
	std::vector<T> data;

public :
	NPVector() {
		data.resize(N);
	}

	T& operator[](size_t i) {
		return data[i];
	}

	void print() {
		for (iter i=data.begin(); i != data.end(); ++i)
			std::cout << *i << " ";
	}
};

typedef NPVector<float, 3> NPVector3f;



#endif /* MYVEC_H_ */
