#ifndef LIBCUMAT_EXPR_H_
#define LIBCUMAT_EXPR_H_

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
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args) const;
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
std::string BinaryOpExpression<Op, Expr1, Expr2>::buildKernel(std::string &params, int &num, std::vector<void *> &args) const
{
	return op_(u_, v_, params, num, args);
}

// -------------- Addition Overloads --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<kernelOp::vectorSum, Expr1, Expr2> operator+(const Expr1 &u, const Expr2 &v)
{
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<kernelOp::vectorSum, Expr1, Expr2>(u, v);
}

template<typename Expr>
const BinaryOpExpression<kernelOp::scalarSum, Expr, double> operator+(const Expr &u, const double &n)
{
	return BinaryOpExpression<kernelOp::scalarSum, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<kernelOp::scalarSum, Expr, double> operator+(const double &n, const Expr &u)
{
	return BinaryOpExpression<kernelOp::scalarSum, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<kernelOp::scalarSum, Expr, float> operator+(const Expr &u, const float &n)
{
	return BinaryOpExpression<kernelOp::scalarSum, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<kernelOp::scalarSum, Expr, float> operator+(const float &n, const Expr &u)
{
	return BinaryOpExpression<kernelOp::scalarSum, Expr, float>(u, n);
}

}

#endif
