#ifndef LIBCUMAT_H_
#define LIBCUMAT_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include "cublas_v2.h"

#include <iostream>
#include <utility>
#include <assert.h>
#include <cstdlib>

template<typename T>
class cumat
{
	private:

	size_t m_rows;
	size_t m_cols;
	thrust::device_vector<T> m_data;

	T generateRandom(const T &min, const T &max);

	public:

	cumat(size_t rows, size_t cols);
	cumat(void);

	size_t rows(void) const;
	size_t cols(void) const;
	size_t size(void) const;

	T get(const size_t &row, const size_t &col) const;
	void set(const size_t &row, const size_t &col, const T &val);

	void fill(const T &val);
	void zero(void);
	void rand(void);
	void rand(const T &min, const T &max);

	//----------------------------------------------
	// Operator Overloads
	//----------------------------------------------

	friend std::ostream& operator<<(std::ostream &os, const cumat &mat)
	{
		const size_t rows = mat.rows();
		const size_t cols = mat.cols();

		for (int i = 0; i < rows; i++) {

			for (int j = 0; j < cols; j++)
				os << mat.get(i, j) << ' ';

			if (i < rows - 1)
				os << "\r\n";
		}

		return os;
	}
};

#endif
