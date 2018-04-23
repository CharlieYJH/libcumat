#include "libcumat_matrix.h"

namespace Cumat
{

cublasHandle_t cublas_handle;
std::unordered_map<std::string, CUmodule> module_cache;

void init(void)
{
	checkCudaErrors(cublasCreate(&Cumat::cublas_handle));
}

void end(void)
{
	checkCudaErrors(cublasDestroy(Cumat::cublas_handle));
	for(std::pair<std::string, CUmodule> iter : module_cache)
		cuModuleUnload(iter.second);
}

//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
template<class F>
void Matrix<T>::elementMathOp(Matrix<T> &src, Matrix<T> &dst, const F &func)
{
	if (src.rows_ == 0 || src.cols_ == 0)
		return;

	thrust::transform(src.data_.begin(), src.data_.end(), dst.data_.begin(), func);
}

//----------------------------------------------
// CUDA Library Wrappers
//----------------------------------------------

template<>
void Matrix<float>::curandGenerateRandom(curandGenerator_t &generator, float *output, size_t size)
{
	CURAND_SAFE_CALL(curandGenerateUniform(generator, output, size));
}

template<>
void Matrix<double>::curandGenerateRandom(curandGenerator_t &generator, double *output, size_t size)
{
	CURAND_SAFE_CALL(curandGenerateUniformDouble(generator, output, size));
}

template<>
void Matrix<float>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const float alpha, const float *in_mat, const float beta, float *out_mat)
{
	checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, &alpha, in_mat, cols, &beta, in_mat, cols, out_mat, rows));
}

template<>
void Matrix<double>::cublasTranspose(cublasHandle_t &handle, const int rows, const int cols, const double alpha, const double *in_mat, const double beta, double *out_mat)
{
	checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, &alpha, in_mat, cols, &beta, in_mat, cols, out_mat, rows));
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

//----------------------------------------------
// Public methods
//----------------------------------------------

template<typename T>
template<typename Expr>
Matrix<T>::Matrix(const Expression<Expr> &rhs):
	rows_(rhs.rows()),
	cols_(rhs.cols()),
	data_(rows_ * cols_)
{
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
	Matrix<T>::assign(*this, rhs);
}

template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols):
	rows_(rows),
	cols_(cols),
	data_(rows_ * cols_)
{
	if (rows == 0 || cols == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, const T val):
	rows_(rows),
	cols_(cols),
	data_(rows_ * cols_, val)
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
	data_(rows_ * cols_)
{
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
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
	size_t vec_size = rows * cols;

	// Resize result matrix if necessary
	if (mat.rows_ != rows || mat.cols_ != cols)
		mat.resize(rows, cols);

	// Push output pointer to argument array
	args.push_back(&mat.data_ptr_);

	// Stores whether the current expression has any transpose sub-expressions
	bool has_transpose_expr = false;

	// Build the parameter list and the evaluation line for the kernel code
	std::string params_line = "(" + TypeString<T>::type + " *out";
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

	// std::cout << params_line << std::endl;
	// std::cout << eval_line << std::endl;

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

		if (compileResult != NVRTC_SUCCESS) {
			std::cout << log << '\n';
			delete[] log;
			exit(1);
		}

		delete[] log;

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

template<typename T>
std::string Matrix<T>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
{
	std::string id_num = std::to_string(num++);
	params += (", " + Cumat::TypeString<T>::type + " *v" + id_num);
	args.push_back((void *)&data_ptr_);
	return "v" + id_num + ((transpose) ? "[x * rows + y]" : "[y * cols + x]");
}

template<typename T>
const Matrix<T>& Matrix<T>::eval(void) const
{
	return *this;
}

template<typename T>
const thrust::device_vector<T>& Matrix<T>::thrustVector(void) const
{
	return data_;
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
	if (rows_ == rows && cols_ == cols) return;

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
	std::swap(data_ptr_, mat.data_ptr_);
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

	*this = (*this) * (max - min) + min;
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
void Matrix<T>::transpose(void)
{
	thrust::device_vector<T> temp(cols_ * rows_);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(data_.data());
	T *B = thrust::raw_pointer_cast(temp.data());

	Matrix<T>::cublasTranspose(Cumat::cublas_handle, rows_, cols_, alpha, A, beta, B);

	data_.swap(temp);
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
	this->resize(cols_, rows_);
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

	Matrix<T>::cublasTranspose(Cumat::cublas_handle, rows_, cols_, alpha, A, beta, B);

	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::mmul(const Matrix<T> &mat)
{
	assert(cols_ == mat.rows_);

	Matrix<T> outmat(rows_, mat.cols_);

	if (outmat.rows_ == 0 || outmat.cols_ == 0)
		return outmat;

	outmat.mmul(*this, mat, 0);

	return outmat;
}

template<typename T>
Matrix<T>& Matrix<T>::mmul(const Matrix<T> &lhs, const Matrix<T> &rhs, const T beta)
{
	size_t outrows = lhs.rows_;
	size_t outcols = rhs.cols_;

	assert(lhs.cols_ == rhs.rows_ && this != &lhs && this != &rhs);

	if (rows_ != outrows || cols_ != outcols)
		this->resize(outrows, outcols);

	const T *A = thrust::raw_pointer_cast(lhs.data_.data());
	const T *B = thrust::raw_pointer_cast(rhs.data_.data());
	T *C = thrust::raw_pointer_cast(data_.data());
	
	// Use cublas<t>gemm() to perform C = alpha * A * B + beta * C
	// where alpha = 1, A = data_, B = mat, beta = 0, and C = outmat
	Matrix<T>::cublasGemm(Cumat::cublas_handle, rhs.cols_, lhs.rows_, lhs.cols_, 1.0, B, rhs.cols_, A, lhs.cols_, beta, C, rhs.cols_);

	return *this;
}

template<typename T>
T Matrix<T>::sum(void)
{
	return thrust::reduce(data_.begin(), data_.end());
}

template<typename T>
T Matrix<T>::norm(void)
{
	return std::sqrt(thrust::transform_reduce(data_.begin(), data_.end(), MathOp::square<T>(), 0.0f, thrust::plus<T>()));
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
// *this = op(mat)
//----------------------------------------------

template<typename T>
Matrix<T>& Matrix<T>::abs(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::abs<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::inverse(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::inverse<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::clip(Matrix<T> &mat, const T min, const T max)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::clip<T>(min, max));
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::exp<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp10(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::exp10<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp2(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::exp2<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::log<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log1p(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::log1p<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log10(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::log10<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log2(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::log2<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::pow(Matrix<T> &mat, const T n)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::pow<T>(n));
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::square(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::square<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sqrt(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::sqrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rsqrt(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::rsqrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cube(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::cube<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cbrt(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::cbrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rcbrt(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::rcbrt<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sin(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::sin<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cos(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::cos<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tan(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::tan<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asin(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::asin<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acos(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::acos<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atan(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::atan<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sinh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::sinh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cosh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::cosh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tanh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::tanh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asinh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::asinh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acosh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::acosh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atanh(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::atanh<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sigmoid(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::sigmoid<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ceil(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::ceil<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::floor(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::floor<T>());
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::round(Matrix<T> &mat)
{
	Matrix<T>::elementMathOp(mat, *this, MathOp::round<T>());
	return *this;
}

//----------------------------------------------
// In-Place Element-Wise Math Operations
// *this = op(*this)
//----------------------------------------------

template<typename T>
Matrix<T>& Matrix<T>::abs(void)
{
	this->abs(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::inverse(void)
{
	this->inverse(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::clip(const T min, const T max)
{
	this->clip(*this, min, max);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp(void)
{
	this->exp(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp10(void)
{
	this->exp10(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp2(void)
{
	this->exp2(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log(void)
{
	this->log(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log1p(void)
{
	this->log1p(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log10(void)
{
	this->log10(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log2(void)
{
	this->log2(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::pow(const T n)
{
	this->pow(*this, n);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::square(void)
{
	this->square(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sqrt(void)
{
	this->sqrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rsqrt(void)
{
	this->rsqrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cube(void)
{
	this->cube(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cbrt(void)
{
	this->cbrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rcbrt(void)
{
	this->rcbrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sin(void)
{
	this->sin(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cos(void)
{
	this->cos(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tan(void)
{
	this->tan(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asin(void)
{
	this->asin(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acos(void)
{
	this->acos(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atan(void)
{
	this->atan(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sinh(void)
{
	this->sinh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cosh(void)
{
	this->cosh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tanh(void)
{
	this->tanh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asinh(void)
{
	this->asinh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acosh(void)
{
	this->acosh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atanh(void)
{
	this->atanh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sigmoid(void)
{
	this->sigmoid(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ceil(void)
{
	this->ceil(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::floor(void)
{
	this->floor(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::round(void)
{
	this->round(*this);
	return *this;
}

//----------------------------------------------
// Operator Overloads
//----------------------------------------------

// -------------- Assignment --------------

template<typename T>
template<typename Expr>
Matrix<T>& Matrix<T>::operator=(const Expression<Expr> &rhs)
{
	Matrix<T>::assign(*this, rhs);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &rhs)
{
	if (&rhs == this)
		return *this;

	Matrix<T>::assign(*this, rhs);
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

// -------------- Addition --------------

template<typename T>
template<typename OtherT>
Matrix<T>& Matrix<T>::operator+=(const OtherT &rhs)
{
	*this = *this + rhs;
	return *this;
}

// -------------- Subtraction --------------

template<typename T>
template<typename OtherT>
Matrix<T>& Matrix<T>::operator-=(const OtherT &rhs)
{
	*this = *this - rhs;
	return *this;
}

// -------------- Multiplication (element-wise) --------------

template<typename T>
template<typename OtherT>
Matrix<T>& Matrix<T>::operator*=(const OtherT &rhs)
{
	*this = *this * rhs;
	return *this;
}

// -------------- Division (element-wise) --------------

template<typename T>
template<typename OtherT>
Matrix<T>& Matrix<T>::operator/=(const OtherT &rhs)
{
	*this = *this / rhs;
	return *this;
}

};
