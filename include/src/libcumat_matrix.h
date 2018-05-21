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
#include <thrust/copy.h>
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
#include <type_traits>

#include "libcumat_expression.h"
#include "libcumat_math.h"
#include "libcumat_typestring.h"
#include "libcumat_cudahandler.h"
#include "libcumat_reference.h"
#include "libcumat_constreference.h"
#include "util/identity_matrix.h"

namespace Cumat
{
	template<typename T>
	class Matrix : public Expression<Matrix<T>>, private CudaHandler
	{
		private:

		size_t rows_;
		size_t cols_;
		thrust::device_vector<T> data_;
		CUdeviceptr data_ptr_;

		public:

		// Creates a matrix that is the result of the rhs expression
		template<typename Expr>
		Matrix(const Expression<Expr> &rhs);

		// Creates a matrix that is a copy of the rhs matrix
		Matrix(const Matrix<T> &rhs);

		// Constructs a matrix from a matrix of different type
		template<typename OtherT>
		Matrix(const Matrix<OtherT> &rhs);

		// Initiates a row vector from a thrust device_vector of a different type
		template<typename OtherT>
		Matrix(const thrust::device_vector<OtherT> &rhs);

		// Initiates a row vector from a thrust host_vector of a different type
		template<typename OtherT>
		Matrix(const thrust::host_vector<OtherT> &rhs);

		// Initiates a row vector from a std::vector
		template<typename OtherT>
		Matrix(const std::vector<OtherT> &rhs);

		// Initiates a row matrix from iterators
		template<typename InputIterator, typename = typename std::enable_if<!std::is_integral<InputIterator>::value && !std::is_floating_point<InputIterator>::value, void>::type>
		Matrix(InputIterator first, InputIterator last);

		// Creates a rows x cols matrix with all elements initiated as 0
		Matrix(const size_t rows, const size_t cols);

		// Creates a rows x cols matrix with all elements initiated as val
		Matrix(const size_t rows, const size_t cols, const T val);

		// Creates a 0 x 0 matrix (no memory allocation is done)
		Matrix(void);
		
		// Expression assignment
		template<typename Expr>
		static void assign(Matrix<T> &mat, const Expression<Expr> &rhs);

		// Builds the parameters for the kernel code used for lazy evaluation
		std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const;

		// Returns a const reference to the matrix object
		const Matrix<T>& eval(void) const;

		// Returns a reference to the underlying device vector
		thrust::device_vector<T>& thrustVector(void);

		// Returns a const reference to the underlying device vector
		const thrust::device_vector<T>& thrustVector(void) const;

		// Returns the number of rows
		size_t rows(void) const;

		// Retuns the number of columns
		size_t cols(void) const;

		// Returns the total size (rows * columns)
		size_t size(void) const;

		// Resizes the matrix
		void resize(size_t rows, size_t cols);

		// Swaps matrix contents with another
		void swap(Matrix<T> &mat);

		// Fills matrix with a value
		void fill(const T val);

		// Fills matrix with 0
		void zero(void);

		// Copies the contents from the iterators to the underlying vector
		// If the vector size is less than the input size, no dimensions are changed, otherwise it's resized to be a row vector
		template<typename InputIterator>
		void copy(InputIterator first, InputIterator last);

		// Fills matrix with random numbers between min and max
		void rand(const T min = -1.0, const T max = 1.0);

		// Returns a matrix initiated with random values between min and max
		static Matrix<T> random(const size_t rows, const size_t cols, const T min = -1.0, const T max = 1.0);

		// Fills the matrix with the identity matrix
		void identity(void);

		// Returns an identity matrix of size rows x cols
		static Matrix<T> identity(const size_t rows, const size_t cols);

		// Performs in-place transpose of current matrix
		void transpose(void);

		// Transposes mat and places the result in this matrix
		Matrix<T>& transpose(Matrix<T> &mat);

		// Performs the following matrix multiplication: *this = lhs * rhs + beta * (*this)
		Matrix<T>& mmul(const Matrix<T> &lhs, const Matrix<T> &rhs, const T beta = 0);

		// Gives the sum of all elements in the matrix
		T sum(void);

		// Gives the 2-norm of the matrix (Frobenius norm)
		T norm(void);

		// Returns the max element in the matrix
		T maxElement(void);

		// Returns the index of the max element of the matrix
		int maxIndex(void);

		// Returns the min element of the matrix
		T minElement(void);

		// Returns the index of the min element of the matrix
		int minIndex(void);

		//----------------------------------------------
		// In-Place Element-Wise Math Operations
		// *this = op(*this)
		//----------------------------------------------

		Matrix<T>& abs(void);

		Matrix<T>& inverse(void);

		Matrix<T>& clip(T min, T max);

		Matrix<T>& exp(void);

		Matrix<T>& exp10(void);

		Matrix<T>& exp2(void);

		Matrix<T>& log(void);

		Matrix<T>& log1p(void);

		Matrix<T>& log10(void);

		Matrix<T>& log2(void);

		Matrix<T>& pow(const T n);
		
		Matrix<T>& powf(const T n);

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

		Matrix<T>& rint(void);

		//----------------------------------------------
		// Operator Overloads
		//----------------------------------------------
		
		// -------------- Assignment --------------
		template<typename Expr>
		Matrix<T>& operator=(const Expression<Expr> &rhs);

		Matrix<T>& operator=(const Matrix<T> &rhs);

		// -------------- Accessor --------------
		MatrixReference<T> operator()(const size_t row, const size_t col);

		MatrixConstReference<T> operator()(const size_t row, const size_t col) const;

		MatrixReference<T> operator()(const size_t idx);

		MatrixConstReference<T> operator()(const size_t idx) const;

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
		friend std::ostream& operator<<(std::ostream &os, Matrix &mat)
		{
			const size_t rows = mat.rows();
			const size_t cols = mat.cols();

			if (rows == 0 || cols == 0)
				return os;

			for (size_t i = 0; i < rows; i++) {

				for (size_t j = 0; j < cols; j++)
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

#include "libcumat_matrix.inl"

#endif
