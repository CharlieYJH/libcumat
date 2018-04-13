#include "libcumat.h"

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n" , __FILE__ , __LINE__);		 \
    exit(EXIT_FAILURE);}} while(0)

namespace Cumat
{

cublasHandle_t cublas_handle;
std::unordered_map<std::string, char *> kernel_cache;

void init(void)
{
	checkCudaErrors(cublasCreate(&Cumat::cublas_handle));
}

void end(void)
{
	checkCudaErrors(cublasDestroy(Cumat::cublas_handle));
	for(std::pair<std::string, char *> iter : kernel_cache)
		delete iter.second;
}

//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
template<class F>
void Matrix<T>::elementMathOp(Matrix<T> &mat, F func)
{
	if (mat.rows_ == 0 || mat.cols_ == 0)
		return;

	thrust::transform(mat.data_.begin(), mat.data_.end(), mat.data_.begin(), func);
}

template<>
std::string Matrix<float>::type(void) const
{
	return "float";
}

template<>
std::string Matrix<double>::type(void) const
{
	return "double";
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
	rows_(rows),
	cols_(cols),
	data_(rows_ * cols_),
	id_("v")
{
	if (rows == 0 || cols == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
Matrix<T>::Matrix(void):
	rows_(0),
	cols_(0),
	data_(rows_ * cols_),
	id_("v")
{
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
std::string Matrix<T>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
{
	std::string id_num = std::to_string(num++);
	params += (", " + this->type() + " *" + id_ + id_num);
	args.push_back((void *)&data_ptr_);
	return id_ + id_num + ((transpose) ? "[x * rows + y]" : "[y * cols + x]");
}

template<typename T>
const Matrix<T>& Matrix<T>::eval(void) const
{
	return *this;
}

template<typename T>
size_t Matrix<T>::rows(void) const
{
	return rows_;
}

template<typename T>
size_t Matrix<T>::cols(void) const
{
	return cols_;
}

template<typename T>
size_t Matrix<T>::size(void) const
{
	return rows_ * cols_;
}

template<typename T>
void Matrix<T>::resize(size_t rows, size_t cols)
{
	if (rows == 0 || cols == 0) {
		rows = 0;
		cols = 0;
	}

	if (rows_ * cols_ != rows * cols) {
		data_.resize(rows * cols);
		data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
	}

	rows_ = rows;
	cols_ = cols;
}

template<typename T>
void Matrix<T>::set(const size_t row, const size_t col, const T val)
{
	assert(row < rows_ && col < cols_);
	data_[row * cols_ + col] = val;
}

template<typename T>
void Matrix<T>::set(const size_t idx, const T val)
{
	assert(idx < rows_ * cols_);
	data_[idx] = val;
}

template<typename T>
void Matrix<T>::swap(Matrix<T> &mat)
{
	if (&mat == this) return;
	std::swap(rows_, mat.rows_);
	std::swap(cols_, mat.cols_);
	data_.swap(mat.data_);
}

template<typename T>
void Matrix<T>::fill(const T val)
{
	thrust::fill(data_.begin(), data_.end(), val);
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
	Matrix<T>::curandGenerateRandom(prng, thrust::raw_pointer_cast(data_.data()), rows_ * cols_);

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
	Matrix<T> transposed_matrix(cols_, rows_);

	Matrix<T>::transpose(transposed_matrix);

	return transposed_matrix;
}

template<typename T>
Matrix<T>& Matrix<T>::transpose(Matrix<T> &mat)
{
	assert(&mat != this);

	if (mat.rows_ != cols_ && mat.cols_ != rows_)
		mat.resize(cols_, rows_);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(data_.data());
	T *B = thrust::raw_pointer_cast(mat.data_.data());

	Matrix<T>::cublasTranspose(Cumat::cublas_handle, rows_, cols_, &alpha, A, &beta, B);

	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::mmul(const Matrix<T> &mat)
{
	assert(cols_ == mat.rows_);

	Matrix<T> outmat(rows_, mat.cols_);

	if (outmat.rows_ == 0 || outmat.cols_ == 0)
		return outmat;

	Matrix<T>::mmul(mat, outmat);

	return outmat;
}

template<typename T>
Matrix<T>& Matrix<T>::mmul(const Matrix<T> &mat, Matrix<T> &outmat)
{
	assert(cols_ == mat.rows_ && this != &outmat && &mat != &outmat);

	// Resize the output matrix if the dimension doesn't match
	if (outmat.rows_ != rows_ && outmat.cols_ != mat.cols_)
		this->resize(rows_, mat.cols_);

	const T *A = thrust::raw_pointer_cast(data_.data());
	const T *B = thrust::raw_pointer_cast(mat.data_.data());
	T *C = thrust::raw_pointer_cast(outmat.data_.data());

	// Use cublas<t>gemm() to perform C = alpha * A * B + beta * C
	// where alpha = 1, A = data_, B = mat, beta = 0, and C = outmat
	Matrix<T>::cublasGemm(Cumat::cublas_handle, mat.cols_, rows_, cols_, 1.0, B, mat.cols_, A, cols_, 0, C, mat.cols_);

	return outmat;
}

template<typename T>
T Matrix<T>::sum(void)
{
	return thrust::reduce(data_.begin(), data_.end());
}

template<typename T>
T Matrix<T>::norm(void)
{
	const T *X = thrust::raw_pointer_cast(data_.data());
	T result;

	Matrix<T>::cublasNorm(Cumat::cublas_handle, rows_ * cols_, X, 1, &result);

	return result;
}

template<typename T>
T Matrix<T>::maxElement(void)
{
	typename thrust::device_vector<T>::iterator iter = thrust::max_element(data_.begin(), data_.end());
	return *iter;
}

template<typename T>
int Matrix<T>::maxIndex(void)
{
	typename thrust::device_vector<T>::iterator iter = thrust::max_element(data_.begin(), data_.end());
	return iter - data_.begin();
}

template<typename T>
T Matrix<T>::minElement(void)
{
	typename thrust::device_vector<T>::iterator iter = thrust::min_element(data_.begin(), data_.end());
	return *iter;
}

template<typename T>
int Matrix<T>::minIndex(void)
{
	typename thrust::device_vector<T>::iterator iter = thrust::min_element(data_.begin(), data_.end());
	return iter - data_.begin();
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
	if (rows_ == 0 || cols_ == 0)
		return *this;

	return (*this * *this).eval();
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
	if (rows_ == 0 || cols_ == 0)
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
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &rhs)
{
	if (&rhs == this)
		return *this;

	if (rows_ != rhs.rows_ || cols_ != rhs.cols_)
		this->resize(rhs.rows_, rhs.cols_);

	thrust::copy(rhs.data_.begin(), rhs.data_.end(), data_.begin());

	return *this;
}

// -------------- Accessor --------------
template<typename T>
T Matrix<T>::operator()(const size_t row, const size_t col) const
{
	assert(row < rows_ && col < cols_);
	return data_[row * cols_ + col];
}

template<typename T>
T Matrix<T>::operator()(const size_t idx) const
{
	assert(idx < rows_ * cols_);
	return data_[idx];
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

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = val, and y = data_
	Matrix<T>::cublasAxpy(Cumat::cublas_handle, rows_ * cols_, 1.0, scalar, 0, thrust::raw_pointer_cast(data_.data()), 1);

	checkCudaErrors(cudaFree(scalar));

	return *this;
}

// -------------- Matrix Addition --------------
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &rhs)
{
	assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);

	const T *X = thrust::raw_pointer_cast(rhs.data_.data());
	T *Y = raw_pointer_cast(data_.data());

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = 1, x = rhs, and y = data_
	Matrix<T>::cublasAxpy(Cumat::cublas_handle, rows_ * cols_, 1.0, X, 1, Y, 1);

	return *this;
}

// -------------- Scalar Subtraction --------------
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T val)
{
	*this += -val;
	return *this;
}

// -------------- Matrix Subtraction --------------
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &rhs)
{
	assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);

	const T *X = thrust::raw_pointer_cast(rhs.data_.data());
	T *Y = thrust::raw_pointer_cast(data_.data());

	// use cuBLAS saxpy to do y = alpha * x + y where alpha = -1, x = rhs, and y = data_
	Matrix<T>::cublasAxpy(Cumat::cublas_handle, rows_ * cols_, -1.0, X, 1, Y, 1);

	return *this;
}

// -------------- Scalar Multiplication --------------
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T val)
{
	// Use cublas<t>scal to do x = alpha * x where alpha = val and x = data_
	Matrix<T>::cublasScal(Cumat::cublas_handle, rows_ * cols_, val, thrust::raw_pointer_cast(data_.data()), 1);

	return *this;
}

// -------------- Matrix Multiplication (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &rhs)
{
	assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
	thrust::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(), thrust::multiplies<T>());
	return *this;
}

// -------------- Scalar Division (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T val)
{
	*this *= (1.0 / val);
	return *this;
}

// -------------- Matrix Division (element-wise) --------------
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T> &rhs)
{
	assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
	thrust::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(), thrust::divides<T>());
	return *this;
}
};

// Template explicit instantiation
template class Cumat::Matrix<float>;
template class Cumat::Matrix<double>;
