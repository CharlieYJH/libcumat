#ifndef LIBCUMAT_REFERENCE_H_
#define LIBCUMAT_REFERENCE_H_

#include <thrust/device_vector.h>

namespace Cumat
{

template<typename T>
class MatrixReference
{
    private:
    
    thrust::device_vector<T> &data_;
    const size_t idx_;
    
    template<typename OtherT>
    friend class MatrixReference;

    public:

    // Class constructor
    MatrixReference(thrust::device_vector<T> &data, const size_t idx);

    // Assigns the rhs value to the data at idx
    MatrixReference<T>& operator=(const T &rhs);

    // Assigns from another reference of the same type (prevents deleted function error)
    MatrixReference<T>& operator=(const MatrixReference<T> &rhs);

    // Assigns from another reference of a different type
    template<typename OtherT>
    MatrixReference<T>& operator=(const MatrixReference<OtherT> &rhs);

    // Returns a copy of the value stored in the device_vector indicated by idx using a cast
    operator T() const;

    // Returns a pointer pointing to the address of the referenced object (not this reference itself)
    T* operator&(void) const;

    // Returns a copy of the value stored in the device_vector indicated by idx
    T val(void) const;

    // Swaps the contents of the two references
    void swap(MatrixReference<T> rhs);

    // Prefix increment
    MatrixReference<T>& operator++(void);

    // Postfix increment
    T operator++(int);

    // Prefix decrement
    MatrixReference<T>& operator--(void);

    // Postfix decrement
    T operator--(int);

    // Adds rhs to the value referenced
    MatrixReference<T>& operator+=(const T &rhs);

    // Subtracts rhs from the value referenced
    MatrixReference<T>& operator-=(const T &rhs);

    // Multiplies the value referenced with rhs
    MatrixReference<T>& operator*=(const T &rhs);

    // Divides the value referenced with rhs
    MatrixReference<T>& operator/=(const T &rhs);
};

}

#include "libcumat_reference.inl"

#endif
