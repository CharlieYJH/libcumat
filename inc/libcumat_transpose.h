#ifndef LIBCUMAT_TRANSPOSE_H_
#define LIBCUMAT_TRANSPOSE_H_

#include <vector>
#include "libcumat_expression.h"

namespace Cumat
{

template<typename Expr>
class TransposeExpression : public Expression<TransposeExpression<Expr>>
{
	const Expr &u_;

	public:
	TransposeExpression(const Expr &u) : u_(u) {}
	size_t rows(void) const;
	size_t cols(void) const;
	std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const;
};

template<typename Expr>
size_t TransposeExpression<Expr>::rows(void) const
{
	return u_.cols();
}

template<typename Expr>
size_t TransposeExpression<Expr>::cols(void) const
{
	return u_.rows();
}

template<typename Expr>
std::string TransposeExpression<Expr>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
{
	return u_.buildKernel(params, num, args, true);
}

template<typename Expr>
const TransposeExpression<Expr> Expression<Expr>::operator~(void) const
{
	return TransposeExpression<Expr>(*this);
}

}

#endif
