#ifndef LIBCUMAT_CONSTREFERENCE_H_
#error "Don't include libcumat_constreference.inl directly. Include libcumat_matrix.h."
#endif

#ifndef LIBCUMAT_CONSTREFERENCE_INL_
#define LIBCUMAT_CONSTREFERENCE_INL_

#include <thrust/device_vector.h>

namespace Cumat
{

template<typename T>
MatrixConstReference<T>::MatrixConstReference(const thrust::device_vector<T> &data, const size_t idx):
    data_(data),
    idx_(idx)
{}

template<typename T>
MatrixConstReference<T>::operator T() const
{
    return data_[idx_];
}

template<typename T>
T* MatrixConstReference<T>::operator&(void) const
{
    return thrust::raw_pointer_cast(data_.data()) + idx_;
}

template<typename T>
T MatrixConstReference<T>::val(void) const
{
    return data_[idx_];
}

}

#endif
