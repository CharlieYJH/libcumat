#ifndef LIBCUMAT_MATRIX_H_
#define LIBCUMAT_MATRIX_H_

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <curand.h>
#include <nvrtc.h>
#include <cuda.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <unordered_map>

#include "libcumat_expression.h"
#include "libcumat_math.h"
#include "libcumat_typestring.h"

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(EXIT_FAILURE);                                         \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(EXIT_FAILURE);                                         \
    }                                                             \
  } while(0)

namespace Cumat
{
	extern std::unordered_map<std::string, CUmodule> module_cache;
	extern cublasHandle_t cublas_handle;
	void init(void);
	void end(void);

	template<typename T>
	class Matrix : public Expression<Matrix<T>>
	{
		private:

		size_t rows_;
		size_t cols_;
		thrust::device_vector<T> data_;
		CUdeviceptr data_ptr_;

		template<class F>
		void elementMathOp(Matrix<T> &src, Matrix<T> &dst, const F &func);

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

		// Class constructors
		template<typename Expr>
		Matrix(const Expression<Expr> &rhs);
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

		// -------------- Matrix Multiplication --------------
		// Matrix<T> operator^(const Matrix<T> &rhs);

		// -------------- Scalar Addition --------------
		Matrix<T>& operator+=(const T val);

		// -------------- Matrix Addition --------------
		Matrix<T>& operator+=(const Matrix<T> &rhs);

		// -------------- Expression Addition --------------
		template<typename Expr>
		Matrix<T>& operator+=(const Expression<Expr> &rhs);

		// -------------- Scalar Subtraction --------------
		Matrix<T>& operator-=(const T val);

		// -------------- Matrix Subtraction --------------
		Matrix<T>& operator-=(const Matrix<T> &rhs);

		// -------------- Expression Subtraction --------------
		template<typename Expr>
		Matrix<T>& operator-=(const Expression<Expr> &rhs);

		// -------------- Scalar Multiplication --------------
		Matrix<T>& operator*=(const T val);

		// -------------- Matrix Multiplication (element-wise) --------------
		Matrix<T>& operator*=(const Matrix<T> &rhs);

		// -------------- Expression Multiplication (element-wise) --------------
		template<typename Expr>
		Matrix<T>& operator*=(const Expression<Expr> &rhs);

		// -------------- Scalar Division (element-wise) --------------
		Matrix<T>& operator/=(const T val);

		// -------------- Matrix Division (element-wise) --------------
		Matrix<T>& operator/=(const Matrix<T> &rhs);

		// -------------- Expression Division (element-wise) --------------
		template<typename Expr>
		Matrix<T>& operator/=(const Expression<Expr> &rhs);

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

	template<typename T>
	template<typename Expr>
	Matrix<T>::Matrix(const Expression<Expr> &rhs):
		rows_(0),
		cols_(0)
	{
		data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
		Matrix<T>::assign(*this, rhs);
	}

	//----------------------------------------------
	// Expression Assignment Method
	//----------------------------------------------

	template<typename T>
	template<typename Expr>
	void Matrix<T>::assign(Matrix<T> &mat, const Expression<Expr> &rhs)
	{
		const Expr &expr = rhs;
		int start_num = 0;
		std::vector<void *> args;

		size_t rows = expr.rows();
		size_t cols = expr.cols();
		size_t vec_size = rows * cols;

		// Resize result matrix if necessary
		if (mat.rows_ != rows || mat.cols_ != cols)
			mat.resize(rows, cols);

		// Push output pointer to argument array
		args.push_back(&mat.data_ptr_);

		// Stores whether the current expression has any transpose sub-expressions
		bool has_transpose_expr = false;

		// Build the parameter list and the evaluation line for the kernel code
		std::string params_line = "(" + Cumat::TypeString<T>::type + " *out";
		std::string eval_line = expr.buildKernel(params_line, start_num, args, false, has_transpose_expr);

		if (has_transpose_expr) {
			// If there's a transpose somewhere, we can't use a 1D grid to access everything
			args.push_back(&rows);
			args.push_back(&cols);
			params_line += ", size_t rows, size_t cols)";
		} else {
			// Use a 1D grid if there's no transpose for faster performance
			args.push_back(&vec_size);
			params_line += ", size_t vec_size)";
		}

		eval_line += ";";

		std::cout << params_line << std::endl;
		std::cout << eval_line << std::endl;

		// Build the kernel code
		const std::string kernel_code = "                                   \n\
			extern \"C\" __global__                                         \n\
			void cumat_kernel" + params_line + "							\n\
			{                                                               \n\
			  size_t x = blockIdx.x * blockDim.x + threadIdx.x;           	\n" +

		((has_transpose_expr)
		// Make use of 2D grid if there's a transpose somewhere in the expression
		?	 "size_t y = blockIdx.y * blockDim.y + threadIdx.y;           	\n\
			  if (x < cols && y < rows) {                               	\n"

		// Otherwise use 1D grid for faster performance
		:	 "const size_t y = 0;											\n\
			  const size_t cols = 0;										\n\
			  if (x < vec_size) {											\n"
		) +

				"out[y * cols + x] = " + eval_line + "                   	\n\
			  }                                                             \n\
			}                                                               \n";

		CUmodule module;
		CUfunction kernel;

		if (module_cache.find(kernel_code) != module_cache.end()) {

			// If this code was used before, load it from the cache to prevent recompiling
			module = module_cache[kernel_code];

		} else {

			nvrtcProgram prog;
			const char *opts[] = {"--gpu-architecture=compute_30"};
			NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "cumat_kernel.cu", 0, NULL, NULL));
			nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

			// Obtain compilation log from the program.
			size_t logSize;
			NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
			char *log = new char[logSize];
			NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
			// std::cout << log << '\n';
			delete[] log;
			if (compileResult != NVRTC_SUCCESS)
				exit(1);

			// Obtain PTX from the program.
			size_t ptxSize;
			NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
			char *ptx = new char[ptxSize];
			NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
			// Destroy the program.
			NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

			// Load and cache the module
			CUDA_SAFE_CALL(cuModuleLoadData(&module, ptx));
			module_cache[kernel_code] = module;

			delete[] ptx;
		}

		// Calculated necessary number of threads and blocks needed
		const size_t total_threads = 256;

		const size_t num_threads_x = (has_transpose_expr) ? total_threads / 16 : total_threads;
		const size_t num_threads_y = total_threads / num_threads_x;

		const size_t num_blocks_x = (cols + num_threads_x - 1) / (num_threads_x);
		const size_t num_blocks_y = (has_transpose_expr) ? (rows + num_threads_y - 1) / (num_threads_y) : 1;

		// Call the kernel from the module
		CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "cumat_kernel"));
		CUDA_SAFE_CALL(cuLaunchKernel(kernel, num_blocks_x, num_blocks_y, 1, num_threads_x, num_threads_y, 1, 0, NULL, args.data(), 0));
	}

	//----------------------------------------------
	// Expression-related Operator Overloads
	//----------------------------------------------
	
	template<typename T>
	template<typename Expr>
	Matrix<T>& Matrix<T>::operator=(const Expression<Expr> &rhs)
	{
		Matrix<T>::assign(*this, rhs);
		return *this;
	}

	template<typename T>
	template<typename Expr>
	Matrix<T> &Matrix<T>::operator+=(const Expression<Expr> &rhs)
	{
		*this = *this + rhs;
		return *this;
	}

	template<typename T>
	template<typename Expr>
	Matrix<T> &Matrix<T>::operator-=(const Expression<Expr> &rhs)
	{
		*this = *this - rhs;
		return *this;
	}

	template<typename T>
	template<typename Expr>
	Matrix<T> &Matrix<T>::operator*=(const Expression<Expr> &rhs)
	{
		*this = *this * rhs;
		return *this;
	}

	template<typename T>
	template<typename Expr>
	Matrix<T> &Matrix<T>::operator/=(const Expression<Expr> &rhs)
	{
		*this = *this / rhs;
		return *this;
	}

	//----------------------------------------------
	// Expression Evaluation
	//----------------------------------------------

	template<typename Expr>
	template<typename T>
	Matrix<T> Expression<Expr>::eval(void) const
	{
		Matrix<T> mat;
		Matrix<T>::assign(mat, *this);
		return mat;
	}

	typedef Matrix<double> Matrixd;
	typedef Matrix<float> Matrixf;
};

#include "libcumat_matrix.cu"

#endif
