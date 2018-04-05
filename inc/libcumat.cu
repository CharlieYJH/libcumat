#include "libCumat.h"

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n" , __FILE__ , __LINE__);		 \
    exit(EXIT_FAILURE);}} while(0)

//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
T Cumat<T>::generateRandom(const T min, const T max)
{
	assert(max > min);
	T random = ((T) std::rand()) / (T) RAND_MAX;
	T diff = max - min;
	T r = random * diff;
	return min + r;
}

//----------------------------------------------
// CUDA Library Wrappers
//----------------------------------------------

template<>
void Cumat<float>::curandGenerateRandom(curandGenerator_t &generator, float *output, size_t size)
{
	CURAND_CALL(curandGenerateUniform(generator, output, size));
}

template<>
void Cumat<double>::curandGenerateRandom(curandGenerator_t &generator, double *output, size_t size)
{
	CURAND_CALL(curandGenerateUniformDouble(generator, output, size));
}

template<>
void Cumat<float>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const float *alpha, const float *in_mat, const float *beta, float *out_mat)
{
	checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void Cumat<double>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const double *alpha, const double *in_mat, const double *beta, double *out_mat)
{
	checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void Cumat<float>::cublasAxpy(cublasHandle_t &handle, const int size, const float alpha, const float *x, const int incx, float *y, const int incy)
{
	checkCudaErrors(cublasSaxpy(handle, size, &alpha, x, incx, y, incy));
}

template<>
void Cumat<double>::cublasAxpy(cublasHandle_t &handle, const int size, const double alpha, const double *x, const int incx, double *y, const int incy)
{
	checkCudaErrors(cublasDaxpy(handle, size, &alpha, x, incx, y, incy));
}

template<>
void Cumat<float>::cublasScal(cublasHandle_t &handle, const int size, const float alpha, float *x, int incx)
{
	checkCudaErrors(cublasSscal(handle, size, &alpha, x, incx));
}

template<>
void Cumat<double>::cublasScal(cublasHandle_t &handle, const int size, const double alpha, double *x, int incx)
{
	checkCudaErrors(cublasDscal(handle, size, &alpha, x, incx));
}

template<>
void Cumat<float>::cublasGemm(cublasHandle_t &handle, int m, int n, int k, const float alpha, const float *A, int lda, const float *B, int ldb, const float beta, float *C, int ldc)
{
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template<>
void Cumat<double>::cublasGemm(cublasHandle_t &handle, int m, int n, int k, const double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double *C, int ldc)
{
	checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

//----------------------------------------------
// Public methods
//----------------------------------------------

template<typename T>
Cumat<T>::Cumat(size_t rows, size_t cols):
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
Cumat<T>::Cumat(void):
	m_rows(0),
	m_cols(0)
{}

template<typename T>
size_t Cumat<T>::rows(void) const
{
	return m_rows;
}

template<typename T>
size_t Cumat<T>::cols(void) const
{
	return m_cols;
}

template<typename T>
size_t Cumat<T>::size(void) const
{
	return m_rows * m_cols;
}

template<typename T>
void Cumat<T>::set(const size_t row, const size_t col, const T val)
{
	assert(row < m_rows && col < m_cols);
	m_data[row * m_cols + col] = val;
}

template<typename T>
void Cumat<T>::fill(const T val)
{
	thrust::fill(m_data.begin(), m_data.end(), val);
}

template<typename T>
void Cumat<T>::zero(void)
{
	Cumat<T>::fill(0);
}

template<typename T>
void Cumat<T>::rand(const T min, const T max)
{
	assert(max > min);

	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	Cumat<T>::curandGenerateRandom(prng, thrust::raw_pointer_cast(m_data.data()), m_rows * m_cols);

	(*this *= (max - min)) += min;
}

template<typename T>
Cumat<T> Cumat<T>::random(const size_t rows, const size_t cols, const T min, const T max)
{
	assert(max > min);
	Cumat<T> mat(rows, cols);
	mat.rand(min, max);
	return mat;
}

template<typename T>
Cumat<T> Cumat<T>::transpose(void)
{
	Cumat<T> transposed_matrix(m_cols, m_rows);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(m_data.data());
	T *B = thrust::raw_pointer_cast(transposed_matrix.m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Uses cublas<t>geam() to perform C = alpha * A + beta * B where alpha = 1, A is transposed, and beta = 0
	Cumat<T>::cublasTranspose(handle, m_rows, m_cols, &alpha, A, &beta, B);

	checkCudaErrors(cublasDestroy(handle));

	return transposed_matrix;
}

template<typename T>
Cumat<T> Cumat<T>::mmul(const Cumat<T> &mat)
{
	assert(m_cols == mat.m_rows);

	Cumat<T> outmat(m_rows, mat.m_cols);

	if (outmat.m_rows == 0 || outmat.m_cols == 0)
		return outmat;

	const T *A = thrust::raw_pointer_cast(m_data.data());
	const T *B = thrust::raw_pointer_cast(mat.m_data.data());
	T *C = thrust::raw_pointer_cast(outmat.m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Use cublas<t>gemm() to perform C = alpha * A * B + beta * C
	// where alpha = 1, A = m_data, B = mat, beta = 0, and C = outmat
	Cumat<T>::cublasGemm(handle, mat.m_cols, m_rows, m_cols, 1.0, B, mat.m_cols, A, m_cols, 0, C, mat.m_cols);

	checkCudaErrors(cublasDestroy(handle));

	return outmat;
}

//----------------------------------------------
// Operator Overloads
//----------------------------------------------

// -------------- Assignment --------------
template<typename T>
Cumat<T>& Cumat<T>::operator=(Cumat<T> rhs)
{
	m_rows = rhs.m_rows;
	m_cols = rhs.m_cols;
	m_data.swap(rhs.m_data); // Using swap avoids reallocation of data on the device

	return *this;
}

// -------------- Accessor --------------
template<typename T>
T Cumat<T>::operator()(const size_t row, const size_t col) const
{
	assert(row < m_rows && col < m_cols);
	return m_data[row * m_cols + col];
}

template<typename T>
T Cumat<T>::operator()(const size_t idx) const
{
	assert(idx < m_rows * m_cols);
	return m_data[idx];
}

// -------------- Negation --------------
template<typename T>
Cumat<T> Cumat<T>::operator-(void)
{
	Cumat<T> m = *this;
	return (*this *= -1);
}

// -------------- Transpose --------------
template<typename T>
Cumat<T> Cumat<T>::operator~(void)
{
	return (*this).transpose();
}

// -------------- Matrix Multiplication --------------
template<typename T>
Cumat<T> Cumat<T>::operator^(const Cumat<T> &rhs)
{
	return (*this).mmul(rhs);
}

// -------------- Scalar Addition --------------
template<typename T>
Cumat<T>& Cumat<T>::operator+=(const T val)
{
	T *scalar = nullptr;

	// Create a temporary buffer on the device for the single scalar value
	checkCudaErrors(cudaMalloc((void **)&scalar, sizeof(T)));
	checkCudaErrors(cudaMemcpy(scalar, &val, sizeof(T), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = val, and y = m_data
	Cumat<T>::cublasAxpy(handle, m_rows * m_cols, 1.0, scalar, 0, thrust::raw_pointer_cast(m_data.data()), 1);

	checkCudaErrors(cudaFree(scalar));
	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator+(const T val)
{
	Cumat<T> m = *this;
	return (m += val);
}

// -------------- Matrix Addition --------------
template<typename T>
Cumat<T>& Cumat<T>::operator+=(const Cumat<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);

	const T *X = thrust::raw_pointer_cast(rhs.m_data.data());
	T *Y = raw_pointer_cast(m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = rhs, and y = m_data
	Cumat<T>::cublasAxpy(handle, m_rows * m_cols, 1.0, X, 1, Y, 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator+(const Cumat<T> &rhs)
{
	Cumat<T> m = *this;
	return (m += rhs);
}

// -------------- Scalar Subtraction --------------
template<typename T>
Cumat<T>& Cumat<T>::operator-=(const T val)
{
	*this += -val;
	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator-(const T val)
{
	Cumat<T> m = *this;
	return (m -= val);
}

// -------------- Matrix Subtraction --------------
template<typename T>
Cumat<T>& Cumat<T>::operator-=(const Cumat<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);

	const T *X = thrust::raw_pointer_cast(rhs.m_data.data());
	T *Y = thrust::raw_pointer_cast(m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = -1, x = rhs, and y = m_data
	Cumat<T>::cublasAxpy(handle, m_rows * m_cols, -1.0, X, 1, Y, 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator-(const Cumat<T> &rhs)
{
	Cumat<T> m = *this;
	return (m -= rhs);
}

// -------------- Scalar Multiplication --------------
template<typename T>
Cumat<T>& Cumat<T>::operator*=(const T val)
{
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Use cublas<t>scal to do x = alpha * x where alpha = val and x = m_data
	Cumat<T>::cublasScal(handle, m_rows * m_cols, val, thrust::raw_pointer_cast(m_data.data()), 1);

	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator*(const T val)
{
	Cumat<T> m = *this;
	return (m *= val);
}

// -------------- Matrix Multiplication (element-wise) --------------
template<typename T>
Cumat<T>& Cumat<T>::operator*=(const Cumat<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
	thrust::transform(thrust::device, m_data.begin(), m_data.end(), rhs.m_data.begin(), m_data.begin(), thrust::multiplies<T>());
	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator*(const Cumat<T> &rhs)
{
	Cumat<T> m = *this;
	return (m *= rhs);
}

// -------------- Scalar Division (element-wise) --------------
template<typename T>
Cumat<T>& Cumat<T>::operator/=(const T val)
{
	*this *= (1.0 / val);
	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator/(const T val)
{
	Cumat<T> m = *this;
	return (m /= val);
}

// -------------- Matrix Division (element-wise) --------------
template<typename T>
Cumat<T>& Cumat<T>::operator/=(const Cumat<T> &rhs)
{
	assert(m_rows == rhs.m_rows && m_cols == rhs.m_cols);
	thrust::transform(thrust::device, m_data.begin(), m_data.end(), rhs.m_data.begin(), m_data.begin(), thrust::divides<T>());
	return *this;
}

template<typename T>
Cumat<T> Cumat<T>::operator/(const Cumat<T> &rhs)
{
	Cumat<T> m = *this;
	return (m /= rhs);
}

// Template explicit instantiation
template class Cumat<float>;
template class Cumat<double>;
