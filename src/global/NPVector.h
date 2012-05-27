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
#include <algorithm>
#include <functional>

// Simple vector operations
// Define this to get around the major Eigen/NVCC issues
// This is !NOT! an efficient library implementation -- there are
// lots of temporary copies that happen all over the place.

template <typename T, int N>
class NPVector {
	typedef typename std::vector<T>::iterator iter;
	std::vector<T> data;

public :
	NPVector() {
		data.resize(N);
	}

	iter begin() {
		return data.begin();
	}

	iter end() {
		return data.end();
	}

	T& operator[](size_t i) {
		return data[i];
	}

	void print() {
		for (iter i=data.begin(); i != data.end(); ++i)
			std::cout << *i << " ";
	}

	NPVector<T, N> operator+(NPVector<T, N>& x) {
		NPVector<T, N> z;
		std::transform(x.begin(), x.end(), data.begin(), z.begin(), std::plus<T>());
		return z;
	}


	NPVector<T, N> operator-(NPVector<T, N>& x) {
		NPVector<T, N> z;
		std::transform(x.begin(), x.end(), data.begin(), z.begin(), std::minus<T>());
		return z;
	}

	NPVector<T, N> operator*(T x) {
		NPVector<T, N> z;
		iter it, it2;
		for (it= data.begin(), it2=z.begin(); it != data.end(); ++it, ++it2)
			(*it2) = (*it)*x;
		return z;
	}

	NPVector<T,N> operator/(T x) {
		NPVector<T, N> z;
		iter it, it2;
		for (it= data.begin(), it2=z.begin(); it != data.end(); ++it, ++it2)
			(*it2) = (*it)/x;
		return z;
	}

};

typedef NPVector<float, 3> NPVector3f;




#endif /* MYVEC_H_ */
