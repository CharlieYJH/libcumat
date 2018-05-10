#ifndef GPUCOMPARE_HPP_
#define GPUCOMPARE_HPP_

#include <limits>
#include <thrust/transform.h>
#include <thrust/count.h>
#include "Core"

template<typename T, typename OtherT>
struct approx_vector_equals
{
	T eps_;

	approx_vector_equals(T eps) : eps_(eps) {}

	__host__ __device__ bool operator()(const T &a, const OtherT &b)
	{
		T a_abs = abs(a);
		OtherT b_abs = abs(b);
		T diff = abs(a - b);

		if (a == b || diff < eps_)
			return true;
		else
			return (diff <= (eps_ * max(a_abs, b_abs)));
	}
};

template<typename T, typename OtherT>
struct approx_scalar_equals
{
	T eps_;
	OtherT scalar_;

	approx_scalar_equals(OtherT scalar, T eps) : scalar_(scalar), eps_(eps) {}

	__host__ __device__ bool operator()(const T &a)
	{
		T a_abs = abs(a);
		T b_abs = abs(scalar_);
		T diff = abs(a - scalar_);

		if (a == scalar_ || diff < eps_)
			return true;
		else
			return (diff <= (eps_ * max(a_abs, b_abs)));
	}
};

template<typename T, typename OtherT>
bool approxEqual(const Cumat::Matrix<T> &a, const Cumat::Matrix<OtherT> &b, T eps = std::numeric_limits<float>::epsilon() * 100)
{
	if (a.rows() != b.rows() || a.cols() != b.cols())
		return false;

	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.thrustVector().begin(), a.thrustVector().end(), b.thrustVector().begin(), bool_vec.begin(), approx_vector_equals<T, OtherT>(eps));
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

template<typename T, typename OtherT>
bool approxEqual(const Cumat::Matrix<T> &a, const OtherT &n, T eps = std::numeric_limits<float>::epsilon() * 100)
{
	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.thrustVector().begin(), a.thrustVector().end(), bool_vec.begin(), approx_scalar_equals<T, OtherT>(n, eps));
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

template<typename T, typename OtherT>
bool approxEqual(const thrust::device_vector<T> &a, const thrust::device_vector<OtherT> &b, T eps = std::numeric_limits<float>::epsilon() * 100)
{
	if (a.size() != b.size())
		return false;

	size_t vec_size = a.size();
	thrust::device_vector<bool> bool_vec(vec_size, false);

	thrust::transform(a.begin(), a.end(), b.begin(), bool_vec.begin(), approx_vector_equals<T, OtherT>());
	size_t result = thrust::count(bool_vec.begin(), bool_vec.end(), true);

	return result == vec_size;
}

#endif
