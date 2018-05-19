#ifndef LIBCUMAT_MATRIX_H_
#error "Don't include libcumat_matrix.inl directly. Include libcumat_matrix.h."
#endif

#ifndef LIBCUMAT_MATRIX_INL_
#define LIBCUMAT_MATRIX_INL_

namespace Cumat
{

void init(void)
{
	CudaHandler::init();
}

void end(void)
{
	CudaHandler::end();
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
Matrix<T>::Matrix(const Matrix<T> &rhs):
	rows_(rhs.rows()),
	cols_(rhs.cols()),
	data_(rhs.data_)
{
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
template<typename OtherT>
Matrix<T>::Matrix(const Matrix<OtherT> &rhs):
	rows_(rhs.rows()),
	cols_(rhs.cols()),
	data_(rhs.thrustVector())
{
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
template<typename OtherT>
Matrix<T>::Matrix(const thrust::device_vector<OtherT> &rhs):
	rows_(1),
	cols_(rhs.size())
{
	if (rows_ == 0 || cols_ == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ = rhs;
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
template<typename OtherT>
Matrix<T>::Matrix(const thrust::host_vector<OtherT> &rhs):
	rows_(1),
	cols_(rhs.size()),
	data_(rhs)
{
	if (rows_ == 0 || cols_ == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
template<typename OtherT>
Matrix<T>::Matrix(const std::vector<OtherT> &rhs):
	rows_(1),
	cols_(rhs.size()),
	data_(rhs)
{
	if (rows_ == 0 || cols_ == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
}

template<typename T>
template<typename InputIterator, typename>
Matrix<T>::Matrix(InputIterator first, InputIterator last):
	rows_(1),
	cols_(last - first),
	data_(first, last)
{
	if (rows_ == 0 || cols_ == 0) {
		rows_ = 0;
		cols_ = 0;
	}

	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
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

	// No assignment necessary if expression had 0 length
	if (vec_size == 0)
		return;

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
		void cumat_kernel" + params_line + "                            \n\
		{                                                               \n\
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;             \n" +

	((has_transpose_expr)
	// Make use of 2D grid if there's a transpose somewhere in the expression
	?	 "size_t y = blockIdx.y * blockDim.y + threadIdx.y;             \n\
		  if (x < cols && y < rows) {                                   \n"

	// Otherwise use 1D grid for faster performance
	:	 "const size_t y = 0;                                           \n\
		  const size_t cols = 0;                                        \n\
		  if (x < vec_size) {                                           \n"
	) +

			"out[y * cols + x] = " + eval_line + "                      \n\
		  }                                                             \n\
		}                                                               \n";

	// std::cout << kernel_code << std::endl;
	
	CUmodule module;
	CUfunction kernel;

	if (CudaHandler::module_cache.find(kernel_code) != CudaHandler::module_cache.end()) {

		// If this code was used before, load it from the cache to prevent recompiling
		module = CudaHandler::module_cache[kernel_code];

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
		CudaHandler::module_cache[kernel_code] = module;

		delete[] ptx;
	}

	// Calculated necessary number of threads and blocks needed
	const unsigned int total_threads = 256;

	const unsigned int num_threads_x = (has_transpose_expr) ? total_threads / 16 : total_threads;
	const unsigned int num_threads_y = total_threads / num_threads_x;

	const unsigned int num_blocks_x = (has_transpose_expr) ? (cols + num_threads_x - 1) / (num_threads_x) : (vec_size + num_threads_x - 1) / (num_threads_x);
	const unsigned int num_blocks_y = (has_transpose_expr) ? (rows + num_threads_y - 1) / (num_threads_y) : 1;

	// Call the kernel from the module
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "cumat_kernel"));
	CUDA_SAFE_CALL(cuLaunchKernel(kernel, num_blocks_x, num_blocks_y, 1, num_threads_x, num_threads_y, 1, 0, CudaHandler::curr_stream, args.data(), 0));
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
thrust::device_vector<T>& Matrix<T>::thrustVector(void)
{
	return data_;
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
template<typename InputIterator>
void Matrix<T>::copy(InputIterator first, InputIterator last)
{
	size_t size = last - first;

	// If the input size is greater than the current dimensions, resize to a row vector
	if (size > rows_ * cols_)
		this->resize(1, size);

	thrust::copy(first, last, data_.begin());
}

template<typename T>
void Matrix<T>::rand(const T min, const T max)
{
	assert(max > min);
	CudaHandler::curandGenerateRandom<T>(thrust::raw_pointer_cast(data_.data()), rows_ * cols_);
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
void Matrix<T>::identity(void)
{
	const size_t total_threads = 256;
	const size_t num_threads_x = 16;
	const size_t num_threads_y = total_threads / num_threads_x;

	const size_t num_blocks_x = (cols_ + num_threads_x - 1) / num_threads_x;
	const size_t num_blocks_y = (rows_ + num_threads_y - 1) / num_threads_y;

	dim3 blockDims(num_threads_x, num_threads_y);
	dim3 gridDims(num_blocks_x, num_blocks_y);
	CudaKernel::identityMatrix<T><<<gridDims, blockDims>>>(thrust::raw_pointer_cast(data_.data()), rows_, cols_);
}

template<typename T>
Matrix<T> Matrix<T>::identity(const size_t rows, const size_t cols)
{
	Matrix<T> mat(rows, cols);
	mat.identity();
	return mat;
}

template<typename T>
void Matrix<T>::transpose(void)
{
	if (rows_ == 0 || cols_ == 0)
		return;

	// If it's a column or row vector, just swap the two properties, no need for memory allocation
	if (rows_ == 1 || cols_ == 1) {
		std::swap(rows_, cols_);
		return;
	}

	thrust::device_vector<T> temp(cols_ * rows_);

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(data_.data());
	T *B = thrust::raw_pointer_cast(temp.data());

	CudaHandler::cublasTranspose<T>(rows_, cols_, alpha, A, beta, B);

	data_.swap(temp);
	data_ptr_ = (CUdeviceptr)thrust::raw_pointer_cast(data_.data());
	this->resize(cols_, rows_);
}

template<typename T>
Matrix<T>& Matrix<T>::transpose(Matrix<T> &mat)
{
	assert(&mat != this);

	if (mat.rows_ != cols_ && mat.cols_ != rows_)
		this->resize(mat.cols_, mat.rows_);

	if (rows_ == 0 || cols_ == 0)
		return *this;

	T alpha = 1.0;
	T beta = 0;

	T *A = thrust::raw_pointer_cast(data_.data());
	T *B = thrust::raw_pointer_cast(mat.data_.data());

	CudaHandler::cublasTranspose<T>(mat.rows_, mat.cols_, alpha, B, beta, A);

	return *this;
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
	CudaHandler::cublasGemm<T>(rhs.cols_, lhs.rows_, lhs.cols_, 1.0, B, rhs.cols_, A, lhs.cols_, beta, C, rhs.cols_);

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
// In-Place Element-Wise Math Operations
// *this = op(*this)
//----------------------------------------------

template<typename T>
Matrix<T>& Matrix<T>::abs(void)
{
	*this = Cumat::abs(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::inverse(void)
{
	*this = 1.0f / *this;
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::clip(T min, T max)
{
	if (min > max)
		std::swap(min, max);

	*this = Cumat::maxf(Cumat::minf(*this, max), min);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp(void)
{
	*this = Cumat::exp(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp10(void)
{
	*this = Cumat::exp10(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::exp2(void)
{
	*this = Cumat::exp2(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log(void)
{
	*this = Cumat::log(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log1p(void)
{
	*this = Cumat::log1p(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log10(void)
{
	*this = Cumat::log10(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::log2(void)
{
	*this = Cumat::log2(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::pow(const T n)
{
	*this = Cumat::pow(*this, n);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::powf(const T n)
{
	*this = Cumat::powf(*this, n);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::square(void)
{
	*this = (*this) * (*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sqrt(void)
{
	*this = Cumat::sqrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rsqrt(void)
{
	*this = Cumat::rsqrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cube(void)
{
	*this = (*this) * (*this) * (*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cbrt(void)
{
	*this = Cumat::cbrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rcbrt(void)
{
	*this = Cumat::rcbrt(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sin(void)
{
	*this = Cumat::sin(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cos(void)
{
	*this = Cumat::cos(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tan(void)
{
	*this = Cumat::tan(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asin(void)
{
	*this = Cumat::asin(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acos(void)
{
	*this = Cumat::acos(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atan(void)
{
	*this = Cumat::atan(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sinh(void)
{
	*this = Cumat::sinh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::cosh(void)
{
	*this = Cumat::cosh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::tanh(void)
{
	*this = Cumat::tanh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::asinh(void)
{
	*this = Cumat::asinh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::acosh(void)
{
	*this = Cumat::acosh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::atanh(void)
{
	*this = Cumat::atanh(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::sigmoid(void)
{
	*this = 1.0f / (1.0f + Cumat::exp(-*this));
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::ceil(void)
{
	*this = Cumat::ceil(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::floor(void)
{
	*this = Cumat::floor(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::round(void)
{
	*this = Cumat::round(*this);
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::rint(void)
{
	*this = Cumat::rint(*this);
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
MatrixReference<T> Matrix<T>::operator()(const size_t row, const size_t col)
{
	assert(row < rows_ && col < cols_);
	return MatrixReference<T>(data_, row * cols_ + col);
}

template<typename T>
MatrixConstReference<T> Matrix<T>::operator()(const size_t row, const size_t col) const
{
	assert(row < rows_ && col < cols_);
	return MatrixConstReference<T>(data_, row * cols_ + col);
}

template<typename T>
MatrixReference<T> Matrix<T>::operator()(const size_t idx)
{
	assert(idx < rows_ * cols_);
	return MatrixReference<T>(data_, idx);
}

template<typename T>
MatrixConstReference<T> Matrix<T>::operator()(const size_t idx) const
{
	assert(idx < rows_ * cols_);
	return MatrixConstReference<T>(data_, idx);
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

#endif
