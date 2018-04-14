#ifndef LIBCUMAT_MATH_H_
#define LIBCUMAT_MATH_H_

#include <algorithm>
#include <cmath>

namespace Cumat
{
	namespace MathOp
	{
		template<typename T>
		struct abs
		{
			public:
			abs(void) {}
			__host__ __device__ T operator()(const T val) const { return std::abs(val); }
		};

		template<typename T>
		struct inverse
		{
			public:
			inverse(void) {}
			__host__ __device__ T operator()(const T val) const { return 1.0 / val; }
		};

		template<typename T>
		struct clip
		{
			T min_;
			T max_;
			public:
			clip(const T min, const T max) : min_(std::min(min, max)), max_(std::max(min, max)) {}
			__host__ __device__ T operator()(const T val) const
			{
				return (val < min_) ? min_
									: (val > max_) ? max_
												   : val;
			}
		};

		template<typename T>
		struct exp
		{
			public:
			exp(void) {}
			__host__ __device__ T operator()(const T val) const { return std::exp(val); }
		};

		template<typename T>
		struct exp10
		{
			const T exp_ = 10;
			public:
			exp10(void) {}
			__host__ __device__ T operator()(const T val) const { return std::pow(exp_, val); }
		};

		template<typename T>
		struct exp2
		{
			public:
			exp2(void) {}
			__host__ __device__ T operator()(const T val) const { return std::exp2(val); }
		};

		template<typename T>
		struct log
		{
			public:
			log(void) {}
			__host__ __device__ T operator()(const T val) const { return std::log(val); }
		};

		template<typename T>
		struct log1p
		{
			public:
			log1p(void) {}
			__host__ __device__ T operator()(const T val) const { return std::log1p(val); }
		};

		template<typename T>
		struct log10
		{
			public:
			log10(void) {}
			__host__ __device__ T operator()(const T val) const { return std::log10(val); }
		};

		template<typename T>
		struct log2
		{
			public:
			log2(void) {}
			__host__ __device__ T operator()(const T val) const { return std::log2(val); }
		};

		template<typename T>
		struct pow
		{
			T exp_;
			public:
			pow(const T n) : exp_(n) {}
			__host__ __device__ T operator()(const T val) const { return std::pow(val, exp_); }
		};

		template<typename T>
		struct square
		{
			public:
			square(void) {}
			__host__ __device__ T operator()(const T val) const { return val * val; }
		};

		template<typename T>
		struct sqrt
		{
			public:
			sqrt(void) {}
			__host__ __device__ T operator()(const T val) const { return std::sqrt(val); }
		};

		template<typename T>
		struct rsqrt
		{
			public:
			rsqrt(void) {}
			__host__ __device__ T operator()(const T val) const { return 1.0 / std::sqrt(val); }
		};

		template<typename T>
		struct cube
		{
			public:
			cube(void) {}
			__host__ __device__ T operator()(const T val) const { return val * val * val; }
		};

		template<typename T>
		struct cbrt
		{
			public:
			cbrt(void) {}
			__host__ __device__ T operator()(const T val) const { return std::cbrt(val); }
		};

		template<typename T>
		struct rcbrt
		{
			public:
			rcbrt(void) {}
			__host__ __device__ T operator()(const T val) const { return 1.0 / std::cbrt(val); }
		};

		template<typename T>
		struct sin
		{
			public:
			sin(void) {}
			__host__ __device__ T operator()(const T val) const { return std::sin(val); }
		};

		template<typename T>
		struct cos
		{
			public:
			cos(void) {}
			__host__ __device__ T operator()(const T val) const { return std::cos(val); }
		};

		template<typename T>
		struct tan
		{
			public:
			tan(void) {}
			__host__ __device__ T operator()(const T val) const { return std::tan(val); }
		};

		template<typename T>
		struct asin
		{
			public:
			asin(void) {}
			__host__ __device__ T operator()(const T val) const { return std::asin(val); }
		};

		template<typename T>
		struct acos
		{
			public:
			acos(void) {}
			__host__ __device__ T operator()(const T val) const { return std::acos(val); }
		};

		template<typename T>
		struct atan
		{
			public:
			atan(void) {}
			__host__ __device__ T operator()(const T val) const { return std::atan(val); }
		};

		template<typename T>
		struct sinh
		{
			public:
			sinh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::sinh(val); }
		};

		template<typename T>
		struct cosh
		{
			public:
			cosh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::cosh(val); }
		};

		template<typename T>
		struct tanh
		{
			public:
			tanh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::tanh(val); }
		};

		template<typename T>
		struct asinh
		{
			public:
			asinh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::asinh(val); }
		};

		template<typename T>
		struct acosh
		{
			public:
			acosh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::acosh(val); }
		};

		template<typename T>
		struct atanh
		{
			public:
			atanh(void) {}
			__host__ __device__ T operator()(const T val) const { return std::atanh(val); }
		};

		template<typename T>
		struct sigmoid
		{
			public:
			sigmoid(void) {}
			__host__ __device__ T operator()(const T val) const { return 1.0 / (1 + std::exp(-val)); }
		};

		template<typename T>
		struct ceil
		{
			public:
			ceil(void) {}
			__host__ __device__ T operator()(const T val) const { return std::ceil(val); }
		};

		template<typename T>
		struct floor
		{
			public:
			floor(void) {}
			__host__ __device__ T operator()(const T val) const { return std::floor(val); }
		};

		template<typename T>
		struct round
		{
			public:
			round(void) {}
			__host__ __device__ T operator()(const T val) const { return std::round(val); }
		};
	}
}

#endif
