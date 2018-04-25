#ifndef LIBCUMAT_CUDAHANDLER_H_
#define LIBCUMAT_CUDAHANDLER_H_

#include <cuda.h>
#include <cublas_v2.h>

#include <unordered_map>
#include <string>

#include "util/helper_cuda.h"

#define NVRTC_SAFE_CALL(x)                                        \
	do {                                                          \
		nvrtcResult result = x;                                   \
		if (result != NVRTC_SUCCESS) {                            \
			std::cerr << "\nerror: " #x " failed with error "     \
			<< nvrtcGetErrorString(result) << '\n';           	  \
			exit(EXIT_FAILURE);                                   \
		}                                                         \
	} while(0)

#define CUDA_SAFE_CALL(x)                                         \
	do {                                                          \
		CUresult result = x;                                      \
		if (result != CUDA_SUCCESS) {                             \
			const char *msg;                                      \
			cuGetErrorName(result, &msg);                         \
			std::cerr << "\nerror: " #x " failed with error "     \
			<< msg << '\n';                                   	  \
			exit(EXIT_FAILURE);                                   \
		}                                                         \
	} while(0)

#define CURAND_SAFE_CALL(x)										  \
	do {														  \
		if((x) != CURAND_STATUS_SUCCESS) { 						  \
			printf("Error at %s:%d\n" , __FILE__ , __LINE__);	  \
			exit(EXIT_FAILURE);									  \
		}														  \
	} while(0)

namespace Cumat
{

class CudaHandler
{
	protected:

	static cublasHandle_t cublas_handle;
	static curandGenerator_t curand_prng;
	static std::unordered_map<std::string, CUmodule> module_cache;
	static std::unordered_map<int, cudaStream_t> cuda_stream;
	static cudaStream_t curr_stream;

	public:

	// CUDA default stream
	static const cudaStream_t default_stream;

	// Creates all necessary handles needed for CUDA API calls
	static void init(void);

	// Destroys all CUDA handles and related variables
	static void end(void);

	// cuRAND random generator wrapper
	template<typename T>
	static void curandGenerateRandom(T *output, size_t size);

	// cuBLAS transpose using cublas<t>geam
	template<typename T>
	static void cublasTranspose(const int rows, const int cols, const T alpha, const T *in_mat, const T beta, T *out_mat);

	// cuBLAS matrix multiplication using cublas<t>gemm
	template<typename T>
	static void cublasGemm(int m, int n, int k, const T alpha, const T *A, int lda, const T *B, int ldb, const T beta, T *C, int ldc);

	// Creates the stream and adds it to the map but doesn't set the current stream
	static void createStream(const int stream_num);

	// Creates and set the current stream
	static void setStream(const int stream_num);

	// Resets the stream to the default stream
	static void setDefaultStream(void);

	// Destroys the indicated stream
	static void destroyStream(const int stream_num);
};

//----------------------------------------------
// Static variable definitions
//----------------------------------------------

// cuBLAS handle
cublasHandle_t CudaHandler::cublas_handle;

// cuRAND generator
curandGenerator_t CudaHandler::curand_prng;

// NVRTC kernel module cache
std::unordered_map<std::string, CUmodule> CudaHandler::module_cache;

// CUDA default stream is NULl or 0
const cudaStream_t CudaHandler::default_stream = NULL;

// CUDA streams manager (maps stream # to the stream pointer)
// CUDA default stream is NULL or 0
std::unordered_map<int, cudaStream_t> CudaHandler::cuda_stream({{0, CudaHandler::default_stream}});

// The current stream CUDA is using
cudaStream_t CudaHandler::curr_stream = CudaHandler::default_stream;

//----------------------------------------------
// Public methods
//----------------------------------------------

void CudaHandler::init(void)
{
	// Create cuBLAS handle
	checkCudaErrors(cublasCreate(&cublas_handle));

	// Create and seed cuRAND generator
	CURAND_SAFE_CALL(curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(curand_prng, (unsigned long long)clock()));
}

void CudaHandler::end(void)
{
	// Destroy cuBLAS handle
	checkCudaErrors(cublasDestroy(cublas_handle));

	// Destroy cuRAND generator
	CURAND_SAFE_CALL(curandDestroyGenerator(curand_prng));

	// Unload all NVRTC kernel modules
	for (std::pair<std::string, CUmodule> it : module_cache)
		CUDA_SAFE_CALL(cuModuleUnload(it.second));

	// Destroys all created streams
	for (std::pair<int, cudaStream_t> it : cuda_stream)
		if (it.second != default_stream)
			checkCudaErrors(cudaStreamDestroy(it.second));
}

template<>
void CudaHandler::curandGenerateRandom<float>(float *output, size_t size)
{
	CURAND_SAFE_CALL(curandSetStream(curand_prng, curr_stream));
	CURAND_SAFE_CALL(curandGenerateUniform(curand_prng, output, size));
}

template<>
void CudaHandler::curandGenerateRandom<double>(double *output, size_t size)
{
	CURAND_SAFE_CALL(curandSetStream(curand_prng, curr_stream));
	CURAND_SAFE_CALL(curandGenerateUniformDouble(curand_prng, output, size));
}

template<>
void CudaHandler::cublasTranspose<float>(const int rows, const int cols, const float alpha, const float *in_mat, const float beta, float *out_mat)
{
	checkCudaErrors(cublasSetStream(cublas_handle, curr_stream));
	checkCudaErrors(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, &alpha, in_mat, cols, &beta, in_mat, cols, out_mat, rows));
}

template<>
void CudaHandler::cublasTranspose<double>(const int rows, const int cols, const double alpha, const double *in_mat, const double beta, double *out_mat)
{
	checkCudaErrors(cublasSetStream(cublas_handle, curr_stream));
	checkCudaErrors(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, &alpha, in_mat, cols, &beta, in_mat, cols, out_mat, rows));
}

template<>
void CudaHandler::cublasGemm<float>(int m, int n, int k, const float alpha, const float *A, int lda, const float *B, int ldb, const float beta, float *C, int ldc)
{
	checkCudaErrors(cublasSetStream(cublas_handle, curr_stream));
	checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template<>
void CudaHandler::cublasGemm<double>(int m, int n, int k, const double alpha, const double *A, int lda, const double *B, int ldb, const double beta, double *C, int ldc)
{
	checkCudaErrors(cublasSetStream(cublas_handle, curr_stream));
	checkCudaErrors(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

void CudaHandler::createStream(const int stream_num)
{
	// Stream already exists
	if (cuda_stream.find(stream_num) != cuda_stream.end())
		return;

	cudaStream_t new_stream;
	checkCudaErrors(cudaStreamCreate(&new_stream));
	cuda_stream[stream_num] = new_stream;
}

void CudaHandler::setStream(const int stream_num)
{
	if (cuda_stream.find(stream_num) != cuda_stream.end()) {

		// Use an existing stream pointer if it was created before
		curr_stream = cuda_stream[stream_num];

	} else {

		// Otherwise create a new stream pointer and save it in the manager
		cudaStream_t new_stream;
		checkCudaErrors(cudaStreamCreate(&new_stream));
		cuda_stream[stream_num] = new_stream;
		curr_stream = new_stream;
	}
}

void CudaHandler::setDefaultStream(void)
{
	if (curr_stream != default_stream)
		curr_stream = default_stream;
}

void CudaHandler::destroyStream(const int stream_num)
{
	// Can't destroy default stream
	assert(stream_num != 0);

	if (cuda_stream.find(stream_num) != cuda_stream.end()) {

		// If the stream we're deleting is the current stream, reset current stream to default stream
		if (cuda_stream[stream_num] == curr_stream)
			curr_stream = default_stream;

		// Destroy the stream and erase it from the map
		checkCudaErrors(cudaStreamDestroy(cuda_stream[stream_num]));
		cuda_stream.erase(stream_num);
	}
}

}

#endif
