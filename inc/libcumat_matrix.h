#ifndef LIBCUMAT_MATRIX_H_
#define LIBCUMAT_MATRIX_H_

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nvrtc.h>
#include <cuda.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "libcumat_expression.h"
#include "libcumat_math.h"
#include "libcumat_typestring.h"
#include "libcumat_cudahandler.h"

namespace Cumat
{
	void init(void);
	void end(void);

	template<typename T>
	class Matrix : public Expression<Matrix<T>>, public CudaHandler
	{
		private:

		size_t rows_;
		size_t cols_;
		thrust::device_vector<T> data_;
		CUdeviceptr data_ptr_;

		template<class F>
		void elementMathOp(Matrix<T> &src, Matrix<T> &dst, const F &func);

		public:

		// Class constructors
		template<typename Expr>
		Matrix(const Expression<Expr> &rhs);
		Matrix(const Matrix<T> &rhs);
		Matrix(const size_t rows, const size_t cols);
		Matrix(const size_t rows, const size_t cols, const T val);
		Matrix(void);
		
		// Expression assignment
		template<typename Expr>
		static void assign(Matrix<T> &mat, const Expression<Expr> &rhs);

		// Builds the parameters for the kernel code used for lazy evaluation
		std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const;

		// Returns a const reference to the matrix object
		const Matrix<T>& eval(void) const;

		// Returns a const reference to the underlying device vector
		const thrust::device_vector<T>& thrustVector(void) const;

		// Methods for getting and modifying matrix size information
		size_t rows(void) const;
		size_t cols(void) const;
		size_t size(void) const;
		void resize(size_t rows, size_t cols);

		// Sets matrix elements
		void set(const size_t row, const size_t col, const T val);
		void set(const size_t idx, const T val);

		// Swaps matrix contents with another
		void swap(Matrix<T> &mat);

		// Fills matrix with a value
		void fill(const T val);

		// Fills matrix with 0
		void zero(void);

		// Fills matrix with random numbers between min and max
		void rand(const T min = -1.0, const T max = 1.0);

		// Returns a matrix initiated with random values
		static Matrix<T> random(const size_t rows, const size_t cols, const T min = -1.0, const T max = 1.0);

		// Performs in-place transpose of current matrix
		void transpose(void);

		// Transposes mat and places the result in this matrix
		Matrix<T>& transpose(Matrix<T> &mat);

		// Performs matrix multiplication with mat and returns a new matrix
		Matrix<T> mmul(const Matrix<T> &mat);

		// Performs the following matrix multiplication: *this = lhs * rhs + beta * (*this)
		Matrix<T>& mmul(const Matrix<T> &lhs, const Matrix<T> &rhs, const T beta = 0);

		// Gives the sum of all elements in the matrix
		T sum(void);

		// Gives the 2-norm of the matrix (Frobenius norm)
		T norm(void);

		// Methods for returning max/min elements in the matrix or their index positions
		T maxElement(void);
		int maxIndex(void);
		T minElement(void);
		int minIndex(void);

		//----------------------------------------------
		// Element-Wise Math Operations
		// *this = op(mat)
		//----------------------------------------------

		Matrix<T>& abs(Matrix<T> &mat);
		Matrix<T>& inverse(Matrix<T> &mat);
		Matrix<T>& clip(Matrix<T> &mat, const T min, const T max);

		Matrix<T>& exp(Matrix<T> &mat);
		Matrix<T>& exp10(Matrix<T> &mat);
		Matrix<T>& exp2(Matrix<T> &mat);
		Matrix<T>& log(Matrix<T> &mat);
		Matrix<T>& log1p(Matrix<T> &mat);
		Matrix<T>& log10(Matrix<T> &mat);
		Matrix<T>& log2(Matrix<T> &mat);
		Matrix<T>& pow(Matrix<T> &mat, const T n);
		Matrix<T>& square(Matrix<T> &mat);
		Matrix<T>& sqrt(Matrix<T> &mat);
		Matrix<T>& rsqrt(Matrix<T> &mat);
		Matrix<T>& cube(Matrix<T> &mat);
		Matrix<T>& cbrt(Matrix<T> &mat);
		Matrix<T>& rcbrt(Matrix<T> &mat);

		Matrix<T>& sin(Matrix<T> &mat);
		Matrix<T>& cos(Matrix<T> &mat);
		Matrix<T>& tan(Matrix<T> &mat);
		Matrix<T>& asin(Matrix<T> &mat);
		Matrix<T>& acos(Matrix<T> &mat);
		Matrix<T>& atan(Matrix<T> &mat);
		Matrix<T>& sinh(Matrix<T> &mat);
		Matrix<T>& cosh(Matrix<T> &mat);
		Matrix<T>& tanh(Matrix<T> &mat);
		Matrix<T>& asinh(Matrix<T> &mat);
		Matrix<T>& acosh(Matrix<T> &mat);
		Matrix<T>& atanh(Matrix<T> &mat);

		Matrix<T>& sigmoid(Matrix<T> &mat);
		Matrix<T>& ceil(Matrix<T> &mat);
		Matrix<T>& floor(Matrix<T> &mat);
		Matrix<T>& round(Matrix<T> &mat);

		//----------------------------------------------
		// In-Place Element-Wise Math Operations
		// *this = op(*this)
		//----------------------------------------------

		Matrix<T>& abs(void);
		Matrix<T>& inverse(void);
		Matrix<T>& clip(const T min, const T max);

		Matrix<T>& exp(void);
		Matrix<T>& exp10(void);
		Matrix<T>& exp2(void);
		Matrix<T>& log(void);
		Matrix<T>& log1p(void);
		Matrix<T>& log10(void);
		Matrix<T>& log2(void);
		Matrix<T>& pow(const T n);
		Matrix<T>& square(void);
		Matrix<T>& sqrt(void);
		Matrix<T>& rsqrt(void);
		Matrix<T>& cube(void);
		Matrix<T>& cbrt(void);
		Matrix<T>& rcbrt(void);

		Matrix<T>& sin(void);
		Matrix<T>& cos(void);
		Matrix<T>& tan(void);
		Matrix<T>& asin(void);
		Matrix<T>& acos(void);
		Matrix<T>& atan(void);
		Matrix<T>& sinh(void);
		Matrix<T>& cosh(void);
		Matrix<T>& tanh(void);
		Matrix<T>& asinh(void);
		Matrix<T>& acosh(void);
		Matrix<T>& atanh(void);

		Matrix<T>& sigmoid(void);
		Matrix<T>& ceil(void);
		Matrix<T>& floor(void);
		Matrix<T>& round(void);

		//----------------------------------------------
		// Operator Overloads
		//----------------------------------------------
		
		// -------------- Assignment --------------
		template<typename Expr>
		Matrix<T>& operator=(const Expression<Expr> &rhs);

		Matrix<T>& operator=(const Matrix<T> &rhs);

		// -------------- Accessor --------------
		T operator()(const size_t row, const size_t col) const;

		T operator()(const size_t idx) const;

		// -------------- Addition --------------
		template<typename OtherT>
		Matrix<T>& operator+=(const OtherT &rhs);

		// -------------- Subtraction --------------
		template<typename OtherT>
		Matrix<T>& operator-=(const OtherT &rhs);

		// -------------- Multiplication (element-wise) --------------
		template<typename OtherT>
		Matrix<T>& operator*=(const OtherT &rhs);

		// -------------- Division (element-wise) --------------
		template<typename OtherT>
		Matrix<T>& operator/=(const OtherT &rhs);

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

#include "libcumat_matrix.cu"

#endif
