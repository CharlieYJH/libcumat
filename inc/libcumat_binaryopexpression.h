#ifndef LIBCUMAT_BINARYOPEXPRESSION_H_
#define LIBCUMAT_BINARYOPEXPRESSION_H_

#include "libcumat_expression.h"
#include "libcumat_operators.h"

namespace Cumat
{

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

// Compile time check for whether type T is a numeric type (prevents ambiguous overloads)
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

}

#endif
