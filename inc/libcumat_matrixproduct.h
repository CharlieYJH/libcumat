#ifndef LIBCUMAT_MATRIXPRODUCT_H_
#define LIBCUMAT_MATRIXPRODUCT_H_

#include "libcumat_forwarddeclarations.h"
#include "libcumat_expression.h"

namespace Cumat
{

template<typename Expr1, typename Expr2, typename T>
class MatrixProductExpression : public Expression<MatrixProductExpression<Expr1, Expr2, T>>
{
	const Expr1 &u_;
	const Expr2 &v_;
	mutable Matrix<T> result_;

	public:
	MatrixProductExpression(const Expr1 &u, const Expr2 &v) : u_(u), v_(v) {}
	size_t rows(void) const;
	size_t cols(void) const;
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const;
};

template<typename Expr1, typename Expr2, typename T>
size_t MatrixProductExpression<Expr1, Expr2, T>::rows(void) const
{
	return u_.rows();
}

template<typename Expr1, typename Expr2, typename T>
size_t MatrixProductExpression<Expr1, Expr2, T>::cols(void) const
{
	return v_.cols();
}

template<typename Expr1, typename Expr2, typename T>
std::string MatrixProductExpression<Expr1, Expr2, T>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
{
	result_.mmul(u_, v_, 0);
	return result_.buildKernel(params, num, args, transpose);
}

template<typename Expr1, typename Expr2>
MatrixProductExpression<Expr1, Expr2, float> mmul(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
	const Expr1 &u = lhs;
	const Expr2 &v = rhs;
	assert(u.cols() == v.rows());
	return MatrixProductExpression<Expr1, Expr2, float>(u, v);
}

}

#endif
