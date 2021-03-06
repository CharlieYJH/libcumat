#ifndef LIBCUMAT_EXPR_H_
#define LIBCUMAT_EXPR_H_

#include "libcumat_forwarddeclarations.h"

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

    template<typename T = float>
    Matrix<T> eval(void) const;
};

template<typename Expr>
template<typename T>
Matrix<T> Expression<Expr>::eval(void) const
{
    Matrix<T> mat;
    Matrix<T>::assign(mat, *this);
    return mat;
}

}

#endif
