#include "libcumat.h"

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n" , __FILE__ , __LINE__);		 \
    exit(EXIT_FAILURE);}} while(0)

namespace Cumat
{
//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
template<class F>
void Matrix<T>::elementMathOp(Matrix<T> &mat, F func)
{
	if (mat.m_rows == 0 || mat.m_cols == 0)
		return;

	thrust::transform(mat.m_data.begin(), mat.m_data.end(), mat.m_data.begin(), func);
}

//----------------------------------------------
// CUDA Library Wrappers
//----------------------------------------------

template<>
void Matrix<float>::curandGenerateRandom(curandGenerator_t &generator, float *output, size_t size)
{
	CURAND_CALL(curandGenerateUniform(generator, output, size));
}

template<>
void Matrix<double>::curandGenerateRandom(curandGenerator_t &generator, double *output, size_t size)
{
	CURAND_CALL(curandGenerateUniformDouble(generator, output, size));
}

template<>
void Matrix<float>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const float *alpha, const float *in_mat, const float *beta, float *out_mat)
{
	checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void Matrix<double>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const double *alpha, const double *in_mat, const double *beta, double *out_mat)
{
	checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void Matrix<float>::cublasAxpy(cublasHandle_t &handle, const int size, const float alpha, const float *x, const int incx, float *y, const int incy)
{
	checkCudaErrors(cublasSaxpy(handle, size, &alpha, x, incx, y, incy));
}

template<>
void Matrix<double>::cublasAxpy(cublasHandle_t &handle, const int size, const double alpha, const double *x, const int incx, double *y, const int incy)
{
	checkCudaErrors(cublasDaxpy(handle, size, &alpha, x, incx, y, incy));
}

template<>
void Matrix<float>::cublasScal(cublasHandle_t &handle, const int size, const float alpha, float *x, int incx)
{
	checkCudaErrors(cublasSscal(handle, size, &alpha, x, incx));
}

template<>
void Matrix<double>::cublasScal(cublasHandle_t &handle, const int size, const double alpha, double *x, int incx)
{
	checkCudaErrors(cublasDscal(handle, size, &alpha, x, incx));
}

template<>
void Matrix<float>::cublasGemm(cublasHandle_t &handle, int m, int n, int k, const float alpha, const float *A, int lda, const float *B, int ldb, const float beta, float *C, int ldc)
{
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template<>
void Matrix<double>::cublasGemm(cublasHandle_t &handle, int m, int n, int k, const double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double *C, int ldc)
{
	checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template<>
void Matrix<float>::cublasNorm(cublasHandle_t &handle, int size, const float *x, int incx, float *result)
{
	checkCudaErrors(cublasSnrm2(handle, size, x, incx, result));
}

template<>
void Matrix<double>::cublasNorm(cublasHandle_t &handle, int size, const double *x, int incx, double *result)
{
	checkCudaErrors(cublasDnrm2(handle, size, x, incx, result));
}

//----------------------------------------------
// Public methods
//----------------------------------------------

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols):
	m_rows(rows),
	m_cols(cols)
{
	if (rows == 0 || cols == 0) {
		m_rows = 0;
		m_cols = 0;
	}

	m_data.resize(rows * cols);
}

template<typename T>
Matrix<T>::Matrix(void):
	m_rows(0),
	m_cols(0)
{}

template<typename T>
size_t Matrix<T>::rows(void) const
{
	return m_rows;
}

template<typename T>
size_t Matrix<T>::cols(void) const
{
	return m_cols;
}

template<typename T>
size_t Matrix<T>::size(void) const
{
	return m_rows * m_cols;
}

template<typename T>
void Matrix<T>::set(const size_t row, const size_t col, const T val)
{
	assert(row < m_rows && col < m_cols);
	m_data[row * m_cols + col] = val;
}

template<typename T>
void Matrix<T>::fill(const T val)
{
	thrust::fill(m_data.begin(), m_data.end(), val);
}

template<typename T>
void Matrix<T>::zero(void)
{
	Matrix<T>::fill(0);
}

template<typename T>
void Matrix<T>::rand(const T min, const T max)
{
	assert(max > min);

	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	Matrix<T>::curandGenerateRandom(prng, thrust::raw_pointer_cast(m_data.data()), m_rows * m_cols);

	(*this *= (max - min)) += min;
}

template<typename T>
Matrix<T> Matrix<T>::random(const size_t rows, const size_t cols, const T min, const T max)
{
	assert(max > min);
	Matrix<T> mat(rows, cols);
	mat.rand(min, max);
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::transpose(void)
{
	Matrix<T> transposed_matrix(m_cols, m_rows);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(m_data.data());
	T *B = thrust::raw_pointer_cast(transposed_matrix.m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Uses cublas<t>geam() to perform C = alpha * A + beta * B where alpha = 1, A is transposed, and beta = 0
	Matrix<T>::cublasTranspose(handle, m_rows, m_cols, &alpha, A, &beta, B);

	checkCudaErrors(cublasDestroy(handle));

	return transposed_matrix;
}

template<typename T>
Matrix<T> Matrix<T>::mmul(const Matrix<T> &mat)
{
	assert(m_cols == mat.m_rows);

	Matrix<T> outmat(m_rows, mat.m_cols);

	if (outmat.m_rows == 0 || outmat.m_cols == 0)
		return outmat;

	const T *A = thrust::raw_pointer_cast(m_data.data());
	const T *B = thrust::raw_pointer_cast(mat.m_data.data());
	T *C = thrust::raw_pointer_cast(outmat.m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Use cublas<t>gemm() to perform C = alpha * A * B + beta * C
	// where alpha = 1, A = m_data, B = mat, beta = 0, and C = outmat
	Matrix<T>::cublasGemm(handle, mat.m_cols, m_rows, m_cols, 1.0, B, mat.m_cols, A, m_cols, 0, C, mat.m_cols);

	checkCudaErrors(cublasDestroy(handle));

	return outmat;
}

template<typename T>
T Matrix<T>::sum(void)
{
	return thrust::reduce(m_data.begin(), m_data.end());
}

template<typename T>
T Matrix<T>::norm(void)
{
	const T *X = thrust::raw_pointer_cast(m_data.data());
	T result;

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	Matrix<T>::cublasNorm(handle, m_rows * m_cols, X, 1, &result);

	checkCudaErrors(cublasDestroy(handle));

	return result;
}

//----------------------------------------------
// Element-Wise Math Operations
//----------------------------------------------

template<typename T>
Matrix<T> Matrix<T>::abs(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::abs<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::inverse(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::inverse<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::clip(const T min, const T max)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::clip<T>(min, max));
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::exp(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::exp<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::log(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::log<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::log1p(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::log1p<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::log10(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::log10<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::pow(const T n)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::pow<T>(n));
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::sqrt<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::rsqrt(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::rsqrt<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::square(void)
{
	if (m_rows == 0 || m_cols == 0)
		return *this;
	return (*this) * (*this);
}

template<typename T>
Matrix<T> Matrix<T>::cube(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::cube<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::sin(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::sin<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::cos(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::cos<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::tan(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::tan<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::asin(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::asin<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::acos(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::acos<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::atan(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::atan<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::sinh(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::sinh<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::cosh(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::cosh<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::tanh(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::tanh<T>());
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::sigmoid(void)
{
	Matrix<T> mat = *this;
	Matrix<T>::elementMathOp(mat, MathOp::sigmoid<T>());
	return mat;
}

//----------------------------------------------
// In-Place Element-Wise Math Operations
//----------------------------------------------

template<typename T>
Matrix<T>& Matrix<T>::iabs(void)
{
	elementMathOp(*this, MathOp::abs<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iinverse(void)
{
	elementMathOp(*this, MathOp::inverse<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iclip(const T min, const T max)
{
	elementMathOp(*this, MathOp::clip<T>(min, max));
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iexp(void)
{
	elementMathOp(*this, MathOp::exp<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ilog(void)
{
	elementMathOp(*this, MathOp::log<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ilog1p(void)
{
	elementMathOp(*this, MathOp::log1p<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ilog10(void)
{
	elementMathOp(*this, MathOp::log10<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ipow(const T n)
{
	elementMathOp(*this, MathOp::pow<T>(n));
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::isqrt(void)
{
	elementMathOp(*this, MathOp::sqrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::irsqrt(void)
{
	elementMathOp(*this, MathOp::rsqrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::isquare(void)
{
	if (m_rows == 0 || m_cols == 0)
		return *this;
	return ((*this) *= (*this));
}

template<typename T>
Matrix<T>& Matrix<T>::icube(void)
{
	elementMathOp(*this, MathOp::cube<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::isin(void)
{
	elementMathOp(*this, MathOp::sin<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::icos(void)
{
	elementMathOp(*this, MathOp::cos<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::itan(void)
{
	elementMathOp(*this, MathOp::tan<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iasin(void)
{
	elementMathOp(*this, MathOp::asin<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iacos(void)
{
	elementMathOp(*this, MathOp::acos<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::iatan(void)
{
	elementMathOp(*this, MathOp::atan<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::isinh(void)
{
	elementMathOp(*this, MathOp::sinh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::icosh(void)
{
	elementMathOp(*this, MathOp::cosh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::itanh(void)
{
	elementMathOp(*this, MathOp::tanh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::isigmoid(void)
{
	elementMathOp(*this, MathOp::sigmoid<T>());
	return *this;
}

//----------------------------------------------
// Operator Overloads
//----------------------------------------------

// -------------- Assignment --------------
template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T> rhs)
{
	m_rows = rhs.m_rows;
	m_cols = rhs.m_cols;
	m_data.swap(rhs.m_data); // Using swap avoids reallocation of data on the device

	return *this;
}

// -------------- Accessor --------------
template<typename T>
T Matrix<T>::operator()(const size_t row, const size_t col) const
{
	assert(row < m_rows && col < m_cols);
	return m_data[row * m_cols + col];
}

template<typename T>
T Matrix<T>::operator()(const size_t idx) const
{
	assert(idx < m_rows * m_cols);
	return m_data[idx];
}

// -------------- Negation --------------
template<typename T>
Matrix<T> Matrix<T>::operator-(void)
{
	Matrix<T> m = *this;
	return (*this *= -1);
}

// -------------- Transpose --------------
template<typename T>
Matrix<T> Matrix<T>::operator~(void)
{
	return (*this).transpose();
}

// -------------- Matrix Multiplication --------------
template<typename T>
Matrix<T> Matrix<T>::operator^(const Matrix<T> &rhs)
{
	return (*this).mmul(rhs);
}

// -------------- Scalar Addition --------------
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const T val)
{
	T *scalar = nullptr;

	// Create a temporary buffer on the device for the single scalar value
	checkCudaErrors(cudaMalloc((void **)&scalar, sizeof(T)));
	checkCudaErrors(cudaMemcpy(scalar, &val, sizeof(T), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = val, and y = m_data
	Matrix<T>::cublasAxpy(handle, m_rows * m_cols, 1.0, scalar, 0, thrust::raw_pointer_cast(m_data.data()), 1);

	checkCudaErrors(cudaFree(scalar));
	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const T val)
{
	Matrix<T> m = *this;
	return (m += val);
}

// -------------- Matrix Addition --------------
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);

	const T *X = thrust::raw_pointer_cast(rhs.m_data.data());
	T *Y = raw_pointer_cast(m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = rhs, and y = m_data
	Matrix<T>::cublasAxpy(handle, m_rows * m_cols, 1.0, X, 1, Y, 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs)
{
	Matrix<T> m = *this;
	return (m += rhs);
}

// -------------- Scalar Subtraction --------------
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T val)
{
	*this += -val;
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const T val)
{
	Matrix<T> m = *this;
	return (m -= val);
}

// -------------- Matrix Subtraction --------------
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);

	const T *X = thrust::raw_pointer_cast(rhs.m_data.data());
	T *Y = thrust::raw_pointer_cast(m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = -1, x = rhs, and y = m_data
	Matrix<T>::cublasAxpy(handle, m_rows * m_cols, -1.0, X, 1, Y, 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs)
{
	Matrix<T> m = *this;
	return (m -= rhs);
}

// -------------- Scalar Multiplication --------------
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T val)
{
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Use cublas<t>scal to do x = alpha * x where alpha = val and x = m_data
	Matrix<T>::cublasScal(handle, m_rows * m_cols, val, thrust::raw_pointer_cast(m_data.data()), 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T val)
{
	Matrix<T> m = *this;
	return (m *= val);
}

// -------------- Matrix Multiplication (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
	thrust::transform(m_data.begin(), m_data.end(), rhs.m_data.begin(), m_data.begin(), thrust::multiplies<T>());
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs)
{
	Matrix<T> m = *this;
	return (m *= rhs);
}

// -------------- Scalar Division (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T val)
{
	*this *= (1.0 / val);
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T val)
{
	Matrix<T> m = *this;
	return (m /= val);
}

// -------------- Matrix Division (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
	thrust::transform(m_data.begin(), m_data.end(), rhs.m_data.begin(), m_data.begin(), thrust::divides<T>());
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T> &rhs)
{
	Matrix<T> m = *this;
	return (m /= rhs);
}
};

// Template explicit instantiation
template class Cumat::Matrix<float>;
template class Cumat::Matrix<double>;
