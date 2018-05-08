#ifndef LIBCUMAT_CONSTREFERENCE_H_
#define LIBCUMAT_CONSTREFERENCE_H_

#include <thrust/device_vector.h>

namespace Cumat
{

template<typename T>
class MatrixConstReference
{
	private:
	
	const thrust::device_vector<T> &data_;
	const size_t idx_;

	public:

	// Class constructor
	MatrixConstReference(const thrust::device_vector<T> &data, const size_t idx);

	// Returns a copy of the value stored in the device_vector indicated by idx using a cast
	operator T() const;

	// Returns a pointer pointing to the address of the referenced object (not this reference itself)
	T* operator&(void) const;

	// Returns a copy of the value stored in the device_vector indicated by idx
	T val(void) const;
};

}

#include "libcumat_constreference.inl"

#endif
