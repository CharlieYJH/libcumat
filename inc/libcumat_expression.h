#ifndef LIBCUMAT_EXPR_H_
#define LIBCUMAT_EXPR_H_

#include <type_traits>

#include "libcumat_forwarddeclarations.h"
#include "libcumat_operators.h"

namespace Cumat
{

//----------------------------------------------
// Base Expression Class
//----------------------------------------------

template<typename Expr>
class Expression
{
	public:
	operator const Expr&() const { return static_cast<const Expr &>(*this); }
	size_t rows(void) const { return static_cast<const Expr &>(*this).rows(); }
	size_t cols(void) const { return static_cast<const Expr &>(*this).cols(); };
	const UnaryOpExpression<KernelOp::negative, Expr> operator-(void) const;
	const TransposeExpression<Expr> operator~(void) const;
	template<typename T = double>
	Matrix<T> eval(void) const;
};

//----------------------------------------------
// Binary Operator Overloads
//----------------------------------------------

template<class Op, typename Expr1, typename Expr2>
class BinaryOpExpression: public Expression<BinaryOpExpression<Op, Expr1, Expr2>>
{
	const Expr1 &u_;
	const Expr2 &v_;
	const Op op_;

	public:
	BinaryOpExpression(const Expr1 &u, const Expr2 &v) : u_(u), v_(v) {}
	size_t rows(void) const;
	size_t cols(void) const;
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const;
};

template<class Op, typename Expr1, typename Expr2>
size_t BinaryOpExpression<Op, Expr1, Expr2>::rows(void) const
{
	return u_.rows();
}

template<class Op, typename Expr1, typename Expr2>
size_t BinaryOpExpression<Op, Expr1, Expr2>::cols(void) const
{
	return u_.cols();
}

template<class Op, typename Expr1, typename Expr2>
std::string BinaryOpExpression<Op, Expr1, Expr2>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
{
	return op_(u_, v_, params, num, args, transpose);
}

// -------------- Addition Overloads --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorSum, Expr1, Expr2> operator+(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorSum, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T> operator+(const Expression<Expr> &lhs, const T &n)
{
	return BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T>(lhs, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T> operator+(const T &n, const Expression<Expr> &rhs)
{
	return BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T>(rhs, n);
}

// -------------- Subtraction Overloads --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorSub, Expr1, Expr2> operator-(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorSub, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSubRight<T>, Expr, T> operator-(const Expression<Expr> &lhs, const T &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarSubRight<T>, Expr, T>(u, n); 
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSubLeft<T>, Expr, T> operator-(const T &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarSubLeft<T>, Expr, T>(u, n); 
}

// -------------- Multiplication Overloads --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorMul, Expr1, Expr2> operator*(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorMul, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T> operator*(const Expression<Expr> &lhs, const T &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T> operator*(const T &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T>(u, n);
}

// -------------- Division Overloads --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorDiv, Expr1, Expr2> operator/(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorDiv, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarDivRight<T>, Expr, T> operator/(const Expression<Expr> &lhs, const T &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarDivRight<T>, Expr, T>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarDivLeft<T>, Expr, T> operator/(const T &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarDivLeft<T>, Expr, T>(u, n);
}

// -------------- Powers --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorPow, Expr1, Expr2> pow(const Expression<Expr1> &base, const Expression<Expr2> &exponent)
{
	const Expr1 &u = base;
	const Expr2 &v = exponent;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorPow, Expr1, Expr2>(u, v);
}

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorPowf, Expr1, Expr2> powf(const Expression<Expr1> &base, const Expression<Expr2> &exponent)
{
	const Expr1 &u = base;
	const Expr2 &v = exponent;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorPowf, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarExpPow<T>, Expr, T> pow(const Expression<Expr> &base, const T &exponent)
{
	const Expr &u = base;
	return BinaryOpExpression<KernelOp::scalarExpPow<T>, Expr, T>(u, exponent);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarExpPowf<T>, Expr, T> powf(const Expression<Expr> &base, const T &exponent)
{
	const Expr &u = base;
	return BinaryOpExpression<KernelOp::scalarExpPowf<T>, Expr, T>(u, exponent);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarBasePow<T>, Expr, T> pow(const T &base, const Expression<Expr> &exponent)
{
	const Expr &u = exponent;
	return BinaryOpExpression<KernelOp::scalarBasePow<T>, Expr, T>(u, base);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarBasePowf<T>, Expr, T> powf(const T &base, const Expression<Expr> &exponent)
{
	const Expr &u = exponent;
	return BinaryOpExpression<KernelOp::scalarBasePowf<T>, Expr, T>(u, base);
}

// -------------- Atan2 --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorAtan2, Expr1, Expr2> atan2(const Expression<Expr1> &y, const Expression<Expr2> &x)
{
	const Expr1 &u = y;
	const Expr2 &v = x;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorAtan2, Expr1, Expr2>(u, v);
}

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorAtan2f, Expr1, Expr2> atan2f(const Expression<Expr1> &y, const Expression<Expr2> &x)
{
	const Expr1 &u = y;
	const Expr2 &v = x;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorAtan2f, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Right<T>, Expr, T> atan2(const Expression<Expr> &y, const T &n)
{
	const Expr &u = y;
	return BinaryOpExpression<KernelOp::scalarAtan2Right<T>, Expr, T>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Rightf<T>, Expr, T> atan2f(const Expression<Expr> &y, const T &n)
{
	const Expr &u = y;
	return BinaryOpExpression<KernelOp::scalarAtan2Rightf<T>, Expr, T>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Left<T>, Expr, T> atan2(const T &n, const Expression<Expr> &x)
{
	const Expr &u = x;
	return BinaryOpExpression<KernelOp::scalarAtan2Left<T>, Expr, T>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Leftf<T>, Expr, T> atan2f(const T &n, const Expression<Expr> &x)
{
	const Expr &u = x;
	return BinaryOpExpression<KernelOp::scalarAtan2Leftf<T>, Expr, T>(u, n);
}

//----------------------------------------------
// Unary Operator Overloads
//----------------------------------------------

template<class Op, typename Expr>
class UnaryOpExpression: public Expression<UnaryOpExpression<Op, Expr>>
{
	const Expr &u_;
	const Op op_;

	public:
	UnaryOpExpression(const Expr &u) : u_(u) {}
	UnaryOpExpression(const Expr &u, const Op op) : u_(u), op_(op) {}
	size_t rows(void) const;
	size_t cols(void) const;
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const;
};

template<class Op, typename Expr>
size_t UnaryOpExpression<Op, Expr>::rows(void) const
{
	return u_.rows();
}

template<class Op, typename Expr>
size_t UnaryOpExpression<Op, Expr>::cols(void) const
{
	return u_.cols();
}

template<class Op, typename Expr>
std::string UnaryOpExpression<Op, Expr>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
{
	return op_(u_, params, num, args, transpose);
}

// -------------- Negation Overload --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::negative, Expr> Expression<Expr>::operator-(void) const
{
	return UnaryOpExpression<KernelOp::negative, Expr>(*this);
}

// -------------- Absolute Value --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::abs, Expr> abs(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::abs, Expr>(v);
}

// -------------- Exponentials / Logs --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::exp, Expr> exp(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::exp, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::exp10, Expr> exp10(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::exp10, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::exp2, Expr> exp2(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::exp2, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::log, Expr> log(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::log, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::log1p, Expr> log1p(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::log1p, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::log10, Expr> log10(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::log10, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::log2, Expr> log2(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::log2, Expr>(v);
}

// -------------- Powers / Roots --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::square, Expr> square(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::square, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::sqrt, Expr> sqrt(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::sqrt, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::rsqrt, Expr> rsqrt(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::rsqrt, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::cube, Expr> cube(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::cube, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::cbrt, Expr> cbrt(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::cbrt, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::rcbrt, Expr> rcbrt(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::rcbrt, Expr>(v);
}

// -------------- Trigonometric Functions --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::sin, Expr> sin(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::sin, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::asin, Expr> asin(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::asin, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::sinh, Expr> sinh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::sinh, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::asinh, Expr> asinh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::asinh, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::cos, Expr> cos(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::cos, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::acos, Expr> acos(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::acos, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::cosh, Expr> cosh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::cosh, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::acosh, Expr> acosh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::acosh, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::tan, Expr> tan(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::tan, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::atan, Expr> atan(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::atan, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::tanh, Expr> tanh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::tanh, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::atanh, Expr> atanh(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::atanh, Expr>(v);
}

// -------------- Rounding Functions --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::ceil, Expr> ceil(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::ceil, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::floor, Expr> floor(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::floor, Expr>(v);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::round, Expr> round(const Expression<Expr> &u)
{
	const Expr &v = u;
	return UnaryOpExpression<KernelOp::round, Expr>(v);
}

}

#endif
