#ifndef GPUCOMPARE_HPP_
#define GPUCOMPARE_HPP_

#include <limits>
#include <thrust/transform.h>
#include <thrust/count.h>
#include "Core"

template<typename T>
struct approx_vector_equals
{
	T eps_;
	approx_vector_equals(T eps) : eps_(eps) {}
	__host__ __device__ bool operator()(const T &a, const T &b) { return (abs(a - b) < eps_) || (isnan(a) && isnan(b)); }
};

template<typename T, typename OtherT>
struct approx_scalar_equals
{
	T eps_;
	OtherT scalar_;
	approx_scalar_equals(OtherT scalar, T eps) : scalar_(scalar), eps_(eps) {}
	__host__ __device__ bool operator()(const T &a) { return (abs(a - scalar_) < eps_) || (isnan(a) && isnan(scalar_)); }
};

template<typename T>
bool approxEqual(const Cumat::Matrix<T> &a, const Cumat::Matrix<T> &b, T eps = std::numeric_limits<T>::epsilon() * 100)
{
	if (a.rows() != b.rows() || a.cols() != b.cols())
		return false;

	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.thrustVector().begin(), a.thrustVector().end(), b.thrustVector().begin(), bool_vec.begin(), approx_vector_equals<T>(eps));
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

template<typename T, typename OtherT>
bool approxEqual(const Cumat::Matrix<T> &a, const OtherT &n, T eps = std::numeric_limits<T>::epsilon() * 100)
{
	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.thrustVector().begin(), a.thrustVector().end(), bool_vec.begin(), approx_scalar_equals<T, OtherT>(n, eps));
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

template<typename T>
bool approxEqual(const thrust::device_vector<T> &a, const thrust::device_vector<T> &b)
{
	if (a.size() != b.size())
		return false;

	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.begin(), a.end(), b.begin(), bool_vec.begin(), approx_vector_equals<T>());
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

#endif
