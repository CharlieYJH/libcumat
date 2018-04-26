#ifndef LIBCUMAT_REFERENCE_H_
#error "Don't include libcumat_reference.inl directly. Include libcumat_reference.h."
#endif

#ifndef LIBCUMAT_REFERENCE_INL_
#define LIBCUMAT_REFERENCE_INL_

namespace Cumat
{

template<typename T>
Reference<T>::Reference(thrust::device_vector<T> &data, const size_t idx):
	data_(data),
	idx_(idx)
{}

template<typename T>
T Reference<T>::val(void) const
{
	return data_[idx_];
}

template<typename T>
Reference<T>::operator T() const
{
	return data_[idx_];
}

}

#endif
