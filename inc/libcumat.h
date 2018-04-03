#ifndef LIBCUMAT_H_
#define LIBCUMAT_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <cstdlib>

template<typename T>
class cumat
{
	private:

	size_t m_rows;
	size_t m_cols;
	thrust::device_vector<T> m_data;

	T generateRandom(const T min, const T max);

	//----------------------------------------------
	// cuBLAS Wrappers
	//----------------------------------------------
	void cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const T *alpha, const T *in_mat, const T *beta, T *out_mat);
	void cublasAxpy(cublasHandle_t &handle, const int size, const T alpha, const T *x, const int incx, T *y, const int incy);

	public:

	cumat(size_t rows, size_t cols);
	cumat(void);

	size_t rows(void) const;
	size_t cols(void) const;
	size_t size(void) const;

	T get(const size_t row, const size_t col) const;
	void set(const size_t row, const size_t col, const T val);

	void fill(const T val);
	void zero(void);
	void rand(void);
	void rand(const T min, const T max);

	cumat<T> transpose(void);

	//----------------------------------------------
	// Operator Overloads
	//----------------------------------------------
	
	// -------------- Assignment --------------
	cumat<T>& operator=(cumat<T> rhs);

	// -------------- Addition --------------
	cumat<T>& operator+=(const T val);
	cumat<T> operator+(const T val);

	friend std::ostream& operator<<(std::ostream &os, const cumat &mat)
	{
		const size_t rows = mat.rows();
		const size_t cols = mat.cols();

		for (int i = 0; i < rows; i++) {

			for (int j = 0; j < cols; j++)
				os << std::setw(10) << mat.get(i, j) << ' ';

			if (i < rows - 1)
				os << "\r\n";
		}

		return os;
	}
};

#endif
