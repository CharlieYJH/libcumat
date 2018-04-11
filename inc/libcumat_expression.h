#ifndef LIBCUMAT_EXPR_H_
#define LIBCUMAT_EXPR_H_

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
	const UnaryOpExpression<KernelOp::negative, Expr> operator-(void) const;
	Matrix<double> eval(void) const;
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
const BinaryOpExpression<KernelOp::vectorSum, Expr1, Expr2> operator+(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.rows() == v.rows() && u.cols() == v.cols());
	return BinaryOpExpression<KernelOp::vectorSum, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T>
const BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double> operator+(const Expression<Expr> &lhs, const double &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float> operator+(const Expression<Expr> &lhs, const float &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float>(u, n);
}

template<typename Expr, typename T>
const BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double> operator+(const double &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float> operator+(const float &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float>(u, n);
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

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubRight<double>, Expr, double> operator-(const Expression<Expr> &lhs, const double &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarSubRight<double>, Expr, double>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubRight<float>, Expr, float> operator-(const Expression<Expr> &lhs, const float &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarSubRight<float>, Expr, float>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubLeft<double>, Expr, double> operator-(const double &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarSubLeft<double>, Expr, double>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubLeft<float>, Expr, float> operator-(const float &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarSubLeft<float>, Expr, float>(u, n); 
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

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double> operator*(const Expression<Expr> &lhs, const double &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float> operator*(const Expression<Expr> &lhs, const float &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double> operator*(const double &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float> operator*(const float &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float>(u, n);
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

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivRight<double>, Expr, double> operator/(const Expression<Expr> &lhs, const double &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarDivRight<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivRight<float>, Expr, float> operator/(const Expression<Expr> &lhs, const float &n)
{
	const Expr &u = lhs;
	return BinaryOpExpression<KernelOp::scalarDivRight<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivLeft<double>, Expr, double> operator/(const double &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarDivLeft<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivLeft<float>, Expr, float> operator/(const float &n, const Expression<Expr> &rhs)
{
	const Expr &u = rhs;
	return BinaryOpExpression<KernelOp::scalarDivLeft<float>, Expr, float>(u, n);
}

//----------------------------------------------
// Binary Operator Overloads
//----------------------------------------------

template<class Op, typename Expr>
class UnaryOpExpression: public Expression<UnaryOpExpression<Op, Expr>>
{
	const Expr &u_;
	const Op op_;

	public:
	UnaryOpExpression(const Expr &u) : u_(u) {}
	size_t rows(void) const;
	size_t cols(void) const;
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args) const;
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
std::string UnaryOpExpression<Op, Expr>::buildKernel(std::string &params, int &num, std::vector<void *> &args) const
{
	return op_(u_, params, num, args);
}

template<typename Expr>
const UnaryOpExpression<KernelOp::negative, Expr> Expression<Expr>::operator-(void) const
{
	return UnaryOpExpression<KernelOp::negative, Expr>(*this);
}

}

#endif
