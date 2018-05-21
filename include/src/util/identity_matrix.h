#ifndef LIBCUMAT_IDENTITYMATRIX_H_
#define LIBCUMAT_IDENTITYMATRIX_H_

namespace Cumat
{
namespace CudaKernel
{

template<typename T>
__global__ void identityMatrix(T *mat, const size_t rows, const size_t cols)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
        mat[y * cols + x] = (x == y) ? 1 : 0;
}

}
}

#endif
