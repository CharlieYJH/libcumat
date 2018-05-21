#ifndef LIBCUMAT_UNARYOPEXPRESSION_H_
#define LIBCUMAT_UNARYOPEXPRESSION_H_

#include <vector>

#include "libcumat_expression.h"
#include "libcumat_unaryop.h"

namespace Cumat
{

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

    std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const;
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
std::string UnaryOpExpression<Op, Expr>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
{
    return op_(u_, params, num, args, transpose, has_transpose_expr);
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

template<typename Expr>
const UnaryOpExpression<KernelOp::rint, Expr> rint(const Expression<Expr> &u)
{
    const Expr &v = u;
    return UnaryOpExpression<KernelOp::rint, Expr>(v);
}

// -------------- Miscellaneous Functions --------------

template<typename Expr>
const UnaryOpExpression<KernelOp::sigmoid, Expr> sigmoid(const Expression<Expr> &u)
{
    const Expr &v = u;
    return UnaryOpExpression<KernelOp::sigmoid, Expr>(v);
}

}

#endif
