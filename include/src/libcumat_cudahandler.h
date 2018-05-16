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
	static const cudaStream_t default_stream;

	public:

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

}

#include "libcumat_cudahandler.inl"

#endif
