#ifndef LIBCUMAT_CUDAHANDLER_H_
#error "Don't include libcumat_cudahandler.inl directly. Include libcumat_cudahandler.h."
#endif

#ifndef LIBCUMAT_CUDAHANDLER_INL_
#define LIBCUMAT_CUDAHANDLER_INL_

namespace Cumat
{

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

CudaHandler::CudaHandler(void)
{
    // If this is the first object, initiate all handles and related variables
    if (objectCounter<CudaHandler>::counter_ == 1)
        CudaHandler::init();
}

CudaHandler::~CudaHandler(void)
{
    // If this is the last object, destroy all handles and related variables
    if (objectCounter<CudaHandler>::counter_ == 1)
        CudaHandler::end();
}

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
    std::unordered_map<std::string, CUmodule>::iterator kernel_it = module_cache.begin();
    while (kernel_it != module_cache.end()) {
        CUDA_SAFE_CALL(cuModuleUnload(kernel_it->second));
        kernel_it = module_cache.erase(kernel_it);
    }

    // Destroys all created streams
    std::unordered_map<int, cudaStream_t>::iterator stream_it = cuda_stream.begin();
    while (stream_it != cuda_stream.end()) {
        if (stream_it->second != default_stream) {
            checkCudaErrors(cudaStreamDestroy(stream_it->second));
            stream_it = cuda_stream.erase(stream_it);
        } else {
            ++stream_it;
        }
    }
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
