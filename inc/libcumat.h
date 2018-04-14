#ifndef LIBCUMAT_H_
#define LIBCUMAT_H_

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
#include <chrono>

#include "libcumat_expression.h"
#include "libcumat_transpose.h"
#include "libcumat_math.h"

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
	extern std::unordered_map<std::string, char *> kernel_cache;
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
		std::string id_;

		template<class F>
		void elementMathOp(Matrix<T> &src, Matrix<T> &dst, const F &func);
		const std::string type(void) const;

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

		template<typename Expr>
		Matrix(const Expression<Expr> &rhs);
		Matrix(const size_t rows, const size_t cols);
		Matrix(const size_t rows, const size_t cols, const T val);
		Matrix(void);
		
		template<typename Expr>
		static void assign(Matrix<T> &mat, const Expression<Expr> &rhs);
		std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const;
		const Matrix<T>& eval(void) const;

		size_t rows(void) const;
		size_t cols(void) const;
		size_t size(void) const;
		void resize(size_t rows, size_t cols);

		void set(const size_t row, const size_t col, const T val);
		void set(const size_t idx, const T val);
		void swap(Matrix<T> &mat);
		void fill(const T val);
		void zero(void);
		void rand(const T min = -1.0, const T max = 1.0);

		static Matrix<T> random(const size_t rows, const size_t cols, const T min = -1.0, const T max = 1.0);

		void transpose(void);
		Matrix<T>& transpose(Matrix<T> &mat);

		Matrix<T> mmul(const Matrix<T> &mat);
		Matrix<T>& mmul(const Matrix<T> &lhs, const Matrix<T> &rhs, const T beta);

		T sum(void);
		T norm(void);
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

		// -------------- Scalar Subtraction --------------
		Matrix<T>& operator-=(const T val);

		// -------------- Matrix Subtraction --------------
		Matrix<T>& operator-=(const Matrix<T> &rhs);

		// -------------- Scalar Multiplication --------------
		Matrix<T>& operator*=(const T val);

		// -------------- Matrix Multiplication (element-wise) --------------
		Matrix<T>& operator*=(const Matrix<T> &rhs);

		// -------------- Scalar Division (element-wise) --------------
		Matrix<T>& operator/=(const T val);
		// Matrix<T> operator/(const T val);

		// -------------- Matrix Division (element-wise) --------------
		Matrix<T>& operator/=(const Matrix<T> &rhs);
		// Matrix<T> operator/(const Matrix<T> &rhs);

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
		cols_(0),
		id_("v")
	{
		data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
		Matrix<T>::assign(*this, rhs);
	}

	template<typename T>
	template<typename Expr>
	void Matrix<T>::assign(Matrix<T> &mat, const Expression<Expr> &rhs)
	{
		const Expr &expr = rhs;
		int start_num = 0;
		std::vector<void *> args;

		size_t rows = expr.rows();
		size_t cols = expr.cols();
		// size_t vec_size = rows * cols;

		// Resize result matrix if necessary
		if (mat.rows_ != rows || mat.cols_ != cols)
			mat.resize(rows, cols);

		// Push output pointer to argument array
		args.push_back(&mat.data_ptr_);

		// Build the parameter list and the evaluation line for the kernel code
		std::string params_line = "(" + mat.type() + " *out";
		std::string eval_line = expr.buildKernel(params_line, start_num, args, false);

		// Push the vector size onto the argument array
		args.push_back(&rows);
		args.push_back(&cols);

		params_line += ", size_t rows, size_t cols)";
		eval_line += ";";

		// std::cout << params_line << std::endl;
		// std::cout << eval_line << std::endl;

		// Build the kernel code
		const std::string kernel_code = "                                   \n\
			extern \"C\" __global__                                         \n\
			void cumat_kernel" + params_line + "							\n\
			{                                                               \n\
			  size_t x = blockIdx.x * blockDim.x + threadIdx.x;           	\n\
			  size_t y = blockIdx.y * blockDim.y + threadIdx.y;           	\n\
			  if (x < cols && y < rows) {                               	\n\
				out[y * cols + x] = " + eval_line + "                   	\n\
			  }                                                             \n\
			}                                                               \n";

		char *ptx;

		if (kernel_cache.find(kernel_code) != kernel_cache.end()) {

			// If this code was used before, load it from the cache to prevent recompiling
			ptx = kernel_cache[kernel_code];

		} else {

			nvrtcProgram prog;
			const char *opts[] = {"--gpu-architecture=compute_30", "--fmad=false"};
			NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "cumat_kernel.cu", 0, NULL, NULL));
			nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);

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
			ptx = new char[ptxSize];
			NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
			// Destroy the program.
			NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

			// Cache the ptx
			kernel_cache[kernel_code] = ptx;
		}

		CUmodule module;
		CUfunction kernel;

		// Calculated necessary number of blocks needed
		const size_t num_threads_x = 32;
		const size_t num_threads_y = 32;
		size_t num_blocks_x = (cols + num_threads_x - 1) / (num_threads_x);
		size_t num_blocks_y = (rows + num_threads_y - 1) / (num_threads_y);

		// Call the kernel from the ptx
		CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
		CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "cumat_kernel"));
		CUDA_SAFE_CALL(cuLaunchKernel(kernel, num_blocks_x, num_blocks_y, 1, num_threads_x, num_threads_y, 1, 0, NULL, args.data(), 0));
	}

	template<typename T>
	template<typename Expr>
	Matrix<T>& Matrix<T>::operator=(const Expression<Expr> &rhs)
	{
		Matrix<T>::assign(*this, rhs);
		return *this;
	}

	template<typename Expr>
	Matrix<double> Expression<Expr>::eval(void) const
	{
		Matrix<double> mat;
		Matrix<double>::assign(mat, *this);
		return mat;
	}

	typedef Matrix<double> Matrixd;
	typedef Matrix<float> Matrixf;
};

#endif
