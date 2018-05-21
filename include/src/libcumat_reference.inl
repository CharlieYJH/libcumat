#ifndef LIBCUMAT_REFERENCE_H_
#error "Don't include libcumat_reference.inl directly. Include libcumat_reference.h."
#endif

#ifndef LIBCUMAT_REFERENCE_INL_
#define LIBCUMAT_REFERENCE_INL_

namespace Cumat
{

template<typename T>
MatrixReference<T>::MatrixReference(thrust::device_vector<T> &data, const size_t idx):
    data_(data),
    idx_(idx)
{}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator=(const T &rhs)
{
    data_[idx_] = rhs;
    return *this;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator=(const MatrixReference<T> &rhs)
{
    data_[idx_] = rhs.data_[rhs.idx_];
    return *this;
}

template<typename T>
template<typename OtherT>
MatrixReference<T>& MatrixReference<T>::operator=(const MatrixReference<OtherT> &rhs)
{
    data_[idx_] = rhs.data_[rhs.idx_];
    return *this;
}

template<typename T>
MatrixReference<T>::operator T() const
{
    return data_[idx_];
}

template<typename T>
T* MatrixReference<T>::operator&(void) const
{
    return thrust::raw_pointer_cast(data_.data()) + idx_;
}

template<typename T>
T MatrixReference<T>::val(void) const
{
    return data_[idx_];
}

template<typename T>
void MatrixReference<T>::swap(MatrixReference<T> rhs)
{
    const T temp = data_[idx_];
    data_[idx_] = rhs;
    rhs.data_[rhs.idx_] = temp;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator++(void)
{
    ++data_[idx_];
    return *this;
}

template<typename T>
T MatrixReference<T>::operator++(int)
{
    T result = data_[idx_]++;
    return result;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator--(void)
{
    --data_[idx_];
    return *this;
}

template<typename T>
T MatrixReference<T>::operator--(int)
{
    T result = data_[idx_]--;
    return result;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator+=(const T &rhs)
{
    data_[idx_] += rhs;
    return *this;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator-=(const T &rhs)
{
    data_[idx_] -= rhs;
    return *this;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator*=(const T &rhs)
{
    data_[idx_] *= rhs;
    return *this;
}

template<typename T>
MatrixReference<T>& MatrixReference<T>::operator/=(const T &rhs)
{
    data_[idx_] /= rhs;
    return *this;
}

}

#endif
