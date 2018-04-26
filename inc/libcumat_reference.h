#ifndef LIBCUMAT_REFERENCE_H_
#define LIBCUMAT_REFERENCE_H_

#include <thrust/device_vector.h>

namespace Cumat
{

template<typename T>
class Reference
{
	private:
	
	thrust::device_vector<T> &data_;
	const size_t idx_;

	public:

	// Class constructor
	Reference(thrust::device_vector<T> &data, const size_t idx);

	// Returns a copy of the value stored in the device_vector indicated by idx
	T val(void) const;

	// Returns a copy of the value stored in the device_vector indicated by idx using a cast
	operator T() const;
};

}

#include "libcumat_reference.inl"

#endif
