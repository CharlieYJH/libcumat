#ifndef LIBCumat_H_
#define LIBCumat_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <curand.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <cstdlib>

template<typename T>
class Cumat
{
	private:

	size_t m_rows;
	size_t m_cols;
	thrust::device_vector<T> m_data;

	T generateRandom(const T min, const T max);

	//----------------------------------------------
	// CUDA Library Wrappers
	//----------------------------------------------
	void curandGenerateRandom(curandGenerator_t &generator, T *output, size_t size);
	void cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const T *alpha, const T *in_mat, const T *beta, T *out_mat);
	void cublasAxpy(cublasHandle_t &handle, const int size, const T alpha, const T *x, const int incx, T *y, const int incy);
	void cublasScal(cublasHandle_t &handle, const int size, const T alpha, T *x, const int incx);
	void cublasGemm(cublasHandle_t &handle, int m, int n, int k, const T alpha, const T *A, int lda, const T *B, int ldb, const T beta, T *C, int ldc);

	public:

	Cumat(size_t rows, size_t cols);
	Cumat(void);

	size_t rows(void) const;
	size_t cols(void) const;
	size_t size(void) const;

	T get(const size_t row, const size_t col) const;
	void set(const size_t row, const size_t col, const T val);

	void fill(const T val);
	void zero(void);
	void rand(const T min = -1.0, const T max = 1.0);

	Cumat<T> transpose(void);
	Cumat<T> mmul(const Cumat<T> &mat);

	//----------------------------------------------
	// Operator Overloads
	//----------------------------------------------
	
	// -------------- Assignment --------------
	Cumat<T>& operator=(Cumat<T> rhs);

	// -------------- Negation --------------
	Cumat<T> operator-(void);

	// -------------- Scalar Addition --------------
	Cumat<T>& operator+=(const T val);
	Cumat<T> operator+(const T val);

	// -------------- Matrix Addition --------------
	Cumat<T>& operator+=(const Cumat<T> &rhs);
	Cumat<T> operator+(const Cumat<T> &rhs);

	// -------------- Scalar Subtraction --------------
	Cumat<T>& operator-=(const T val);
	Cumat<T> operator-(const T val);

	// -------------- Matrix Subtraction --------------
	Cumat<T>& operator-=(const Cumat<T> &rhs);
	Cumat<T> operator-(const Cumat<T> &rhs);

	// -------------- Scalar Multiplication --------------
	Cumat<T>& operator*=(const T val);
	Cumat<T> operator*(const T val);

	// -------------- Matrix Multiplication (element-wise) --------------
	Cumat<T>& operator*=(const Cumat<T> &rhs);
	Cumat<T> operator*(const Cumat<T> &rhs);

	// -------------- Scalar Division (element-wise) --------------
	Cumat<T>& operator/=(const T val);
	Cumat<T> operator/(const T val);

	// -------------- Matrix Division (element-wise) --------------
	Cumat<T>& operator/=(const Cumat<T> &rhs);
	Cumat<T> operator/(const Cumat<T> &rhs);

	friend std::ostream& operator<<(std::ostream &os, const Cumat &mat)
	{
		const size_t rows = mat.rows();
		const size_t cols = mat.cols();

		if (rows == 0 || cols == 0)
			return os;

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
