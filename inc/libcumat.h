#ifndef LIBCUMAT_H_
#define LIBCUMAT_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <curand.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <cstdlib>

#include <libcumat_math.h>

namespace Cumat
{
	extern cublasHandle_t cublas_handle;
	void createCublasHandle(void);
	void destroyCublasHandle(void);

	template<typename T>
	class Matrix
	{
		private:

		size_t m_rows;
		size_t m_cols;
		thrust::device_vector<T> m_data;

		template<class F>
		void elementMathOp(Matrix<T> &mat, F func);

		//----------------------------------------------
		// CUDA Library Wrappers
		//----------------------------------------------
		void curandGenerateRandom(curandGenerator_t &generator, T *output, size_t size);
		void cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const T *alpha, const T *in_mat, const T *beta, T *out_mat);
		void cublasAxpy(cublasHandle_t &handle, const int size, const T alpha, const T *x, const int incx, T *y, const int incy);
		void cublasScal(cublasHandle_t &handle, const int size, const T alpha, T *x, const int incx);
		void cublasGemm(cublasHandle_t &handle, int m, int n, int k, const T alpha, const T *A, int lda, const T *B, int ldb, const T beta, T *C, int ldc);
		void cublasNorm(cublasHandle_t &handle, int size, const T *x, int incx, T *result);

		public:

		Matrix(size_t rows, size_t cols);
		Matrix(void);

		size_t rows(void) const;
		size_t cols(void) const;
		size_t size(void) const;

		void set(const size_t row, const size_t col, const T val);
		void set(const size_t idx, const T val);
		void swap(Matrix<T> &mat);
		void fill(const T val);
		void zero(void);
		void rand(const T min = -1.0, const T max = 1.0);

		static Matrix<T> random(const size_t rows, const size_t cols, const T min = -1.0, const T max = 1.0);

		Matrix<T> transpose(void);
		Matrix<T>& transpose(Matrix<T> &mat);

		Matrix<T> mmul(const Matrix<T> &mat);
		Matrix<T>& mmul(const Matrix<T> &mat, Matrix<T> &outmat);

		T sum(void);
		T norm(void);
		T maxElement(void);
		int maxIndex(void);
		T minElement(void);
		int minIndex(void);

		//----------------------------------------------
		// Element-Wise Math Operations
		//----------------------------------------------

		Matrix<T> abs(void);
		Matrix<T> inverse(void);
		Matrix<T> clip(const T min, const T max);

		Matrix<T> exp(void);
		Matrix<T> log(void);
		Matrix<T> log1p(void);
		Matrix<T> log10(void);
		Matrix<T> pow(const T n);
		Matrix<T> sqrt(void);
		Matrix<T> rsqrt(void);
		Matrix<T> square(void);
		Matrix<T> cube(void);

		Matrix<T> sin(void);
		Matrix<T> cos(void);
		Matrix<T> tan(void);
		Matrix<T> asin(void);
		Matrix<T> acos(void);
		Matrix<T> atan(void);
		Matrix<T> sinh(void);
		Matrix<T> cosh(void);
		Matrix<T> tanh(void);

		Matrix<T> sigmoid(void);

		//----------------------------------------------
		// In-Place Element-Wise Math Operations
		//----------------------------------------------

		Matrix<T>& iabs(void);
		Matrix<T>& iinverse(void);
		Matrix<T>& iclip(const T min, const T max);

		Matrix<T>& iexp(void);
		Matrix<T>& ilog(void);
		Matrix<T>& ilog1p(void);
		Matrix<T>& ilog10(void);
		Matrix<T>& ipow(const T n);
		Matrix<T>& isqrt(void);
		Matrix<T>& irsqrt(void);
		Matrix<T>& isquare(void);
		Matrix<T>& icube(void);

		Matrix<T>& isin(void);
		Matrix<T>& icos(void);
		Matrix<T>& itan(void);
		Matrix<T>& iasin(void);
		Matrix<T>& iacos(void);
		Matrix<T>& iatan(void);
		Matrix<T>& isinh(void);
		Matrix<T>& icosh(void);
		Matrix<T>& itanh(void);

		Matrix<T>& isigmoid(void);

		//----------------------------------------------
		// Operator Overloads
		//----------------------------------------------
		
		// -------------- Assignment --------------
		Matrix<T>& operator=(Matrix<T> rhs);

		// -------------- Accessor --------------
		T operator()(const size_t row, const size_t col) const;
		T operator()(const size_t idx) const;
		
		// -------------- Negation --------------
		Matrix<T> operator-(void);

		// -------------- Transpose --------------
		Matrix<T> operator~(void);

		// -------------- Matrix Multiplication --------------
		Matrix<T> operator^(const Matrix<T> &rhs);

		// -------------- Scalar Addition --------------
		Matrix<T>& operator+=(const T val);
		Matrix<T> operator+(const T val);

		// -------------- Matrix Addition --------------
		Matrix<T>& operator+=(const Matrix<T> &rhs);
		Matrix<T> operator+(const Matrix<T> &rhs);

		// -------------- Scalar Subtraction --------------
		Matrix<T>& operator-=(const T val);
		Matrix<T> operator-(const T val);

		// -------------- Matrix Subtraction --------------
		Matrix<T>& operator-=(const Matrix<T> &rhs);
		Matrix<T> operator-(const Matrix<T> &rhs);

		// -------------- Scalar Multiplication --------------
		Matrix<T>& operator*=(const T val);
		Matrix<T> operator*(const T val);

		// -------------- Matrix Multiplication (element-wise) --------------
		Matrix<T>& operator*=(const Matrix<T> &rhs);
		Matrix<T> operator*(const Matrix<T> &rhs);

		// -------------- Scalar Division (element-wise) --------------
		Matrix<T>& operator/=(const T val);
		Matrix<T> operator/(const T val);

		// -------------- Matrix Division (element-wise) --------------
		Matrix<T>& operator/=(const Matrix<T> &rhs);
		Matrix<T> operator/(const Matrix<T> &rhs);

		// -------------- Output Stream Operator --------------
		friend std::ostream& operator<<(std::ostream &os, const Matrix &mat)
		{
			const size_t rows = mat.rows();
			const size_t cols = mat.cols();

			if (rows == 0 || cols == 0)
				return os;

			for (int i = 0; i < rows; i++) {

				for (int j = 0; j < cols; j++)
					os << std::setw(10) << mat(i, j) << ' ';

				if (i < rows - 1)
					os << "\r\n";
			}

			return os;
		}
	};

	typedef Matrix<double> Matrixd;
	typedef Matrix<float> Matrixf;
};

#endif
