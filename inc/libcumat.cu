#include "libcumat.h"

//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
T cumat<T>::generateRandom(const T min, const T max)
{
	assert(max > min);
	T random = ((T) std::rand()) / (T) RAND_MAX;
	T diff = max - min;
	T r = random * diff;
	return min + r;
}

//----------------------------------------------
// cuBLAS Wrappers
//----------------------------------------------

template<>
void cumat<float>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const float *alpha, const float *in_mat, const float *beta, float *out_mat)
{
	checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void cumat<double>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const double *alpha, const double *in_mat, const double *beta, double *out_mat)
{
	checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, alpha, in_mat, cols, beta, in_mat, cols, out_mat, rows));
}

template<>
void cumat<float>::cublasAxpy(cublasHandle_t &handle, const int size, const float alpha, const float *x, const int incx, float *y, const int incy)
{
	checkCudaErrors(cublasSaxpy(handle, size, &alpha, x, incx, y, incy));
}

template<>
void cumat<double>::cublasAxpy(cublasHandle_t &handle, const int size, const double alpha, const double *x, const int incx, double *y, const int incy)
{
	checkCudaErrors(cublasDaxpy(handle, size, &alpha, x, incx, y, incy));
}

//----------------------------------------------
// Public methods
//----------------------------------------------

template<typename T>
cumat<T>::cumat(size_t rows, size_t cols):
	m_rows(rows),
	m_cols(cols)
{
	m_data.resize(rows * cols);
}

template<typename T>
cumat<T>::cumat(void):
	m_rows(0),
	m_cols(0)
{}

template<typename T>
size_t cumat<T>::rows(void) const
{
	return m_rows;
}

template<typename T>
size_t cumat<T>::cols(void) const
{
	return m_cols;
}

template<typename T>
size_t cumat<T>::size(void) const
{
	return m_rows * m_cols;
}

template<typename T>
T cumat<T>::get(const size_t row, const size_t col) const
{
	assert(row < m_rows && col < m_cols);
	return m_data[row * m_cols + col];
}

template<typename T>
void cumat<T>::set(const size_t row, const size_t col, const T val)
{
	assert(row < m_rows && col < m_cols);
	m_data[row * m_cols + col] = val;
}

template<typename T>
void cumat<T>::fill(const T val)
{
	thrust::fill(m_data.begin(), m_data.end(), val);
}

template<typename T>
void cumat<T>::zero(void)
{
	cumat<T>::fill(0);
}

template<typename T>
void cumat<T>::rand()
{
	thrust::host_vector<T> vec(m_data.size());

	for (int i = 0; i < vec.size(); i++)
		vec[i] = cumat<T>::generateRandom(-1, 1);

	m_data = vec;
}

template<typename T>
void cumat<T>::rand(const T min, const T max)
{
	thrust::host_vector<T> vec(m_data.size());

	// Fills the matrix with random numbers between min and max serially
	for (int i = 0; i < vec.size(); i++)
		vec[i] = cumat<T>::generateRandom(min, max);

	m_data = vec;
}

template<typename T>
cumat<T> cumat<T>::transpose(void)
{
	cumat<T> transposed_matrix(m_cols, m_rows);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(m_data.data());
	T *B = thrust::raw_pointer_cast(transposed_matrix.m_data.data());

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// Uses cublas<t>geam() to perform C = alpha * A + beta * B where alpha = 1, A is transposed, and beta = 0
	cumat<T>::cublasTranspose(handle, m_rows, m_cols, &alpha, A, &beta, B);

	checkCudaErrors(cublasDestroy(handle));

	return transposed_matrix;
}

//----------------------------------------------
// Operator Overloads
//----------------------------------------------

// -------------- Assignment --------------
template<typename T>
cumat<T>& cumat<T>::operator=(cumat<T> rhs)
{
	m_rows = rhs.m_rows;
	m_cols = rhs.m_cols;
	m_data.swap(rhs.m_data); // Using swap avoids reallocation of data on the device

	return *this;
}

// -------------- Addition --------------
template<typename T>
cumat<T>& cumat<T>::operator+=(const T val)
{
	T *scalar = nullptr;

	// Create a temporary buffer on the device for the single scalar value
	checkCudaErrors(cudaMalloc((void **)&scalar, sizeof(T)));
	checkCudaErrors(cudaMemcpy(scalar, &val, sizeof(T), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = val, and y = m_data
	cumat<T>::cublasAxpy(handle, m_rows * m_cols, 1.0, scalar, 0, thrust::raw_pointer_cast(m_data.data()), 1);

	checkCudaErrors(cudaFree(scalar));
	checkCudaErrors(cublasDestroy(handle));

	return *this;
}

template<typename T>
cumat<T> cumat<T>::operator+(const T val)
{
	cumat<T> m = *this;
	return (m += val);
}

// Template explicit instantiation
template class cumat<float>;
template class cumat<double>;
