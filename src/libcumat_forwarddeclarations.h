#ifndef LIBCUMAT_FORWARDDECLARATIONS_H_
#define LIBCUMAT_FORWARDDECLARATIONS_H_

namespace Cumat
{

template<typename T> class Matrix;
template<typename Expr> class Expression;
template<class Op, typename Expr1, typename Expr2> class BinaryOpExpression;
template<class Op, typename Expr> class UnaryOpExpression;
template<typename Expr> class TransposeExpression;
template<typename Expr1, typename Expr2, typename T> class MatrixProductExpression;

namespace KernelOp
{

class UnaryOp;
struct negative;
struct abs;
struct exp;
struct exp10;
struct exp2;
struct log;
struct log1p;
struct log10;
struct log2;
struct sqrt;
struct rsqrt;
struct cbrt;
struct rcbrt;
struct sin;
struct asin;
struct sinh;
struct asinh;
struct cos;
struct acos;
struct cosh;
struct acosh;
struct tan;
struct atan;
struct tanh;
struct atanh;
struct ceil;
struct floor;
struct round;
struct rint;

template<typename T> class BinaryScalarOp;
template<typename T> class BinaryScalarOpRight;
template<typename T> struct scalarExpPow;
template<typename T> struct scalarExpPowf;
template<typename T> struct scalarMaxRightf;
template<typename T> struct scalarMinRightf;

} // End of KernelOp namespace

template<typename Expr>
const UnaryOpExpression<KernelOp::abs, Expr> abs(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::exp, Expr> exp(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::exp10, Expr> exp10(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::exp2, Expr> exp2(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::log, Expr> log(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::log1p, Expr> log1p(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::log10, Expr> log10(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::log2, Expr> log2(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::sqrt, Expr> sqrt(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::rsqrt, Expr> rsqrt(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::cbrt, Expr> cbrt(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::rcbrt, Expr> rcbrt(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::sin, Expr> sin(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::asin, Expr> asin(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::sinh, Expr> sinh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::asinh, Expr> asinh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::cos, Expr> cos(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::acos, Expr> acos(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::cosh, Expr> cosh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::acosh, Expr> acosh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::tan, Expr> tan(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::atan, Expr> atan(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::tanh, Expr> tanh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::atanh, Expr> atanh(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::ceil, Expr> ceil(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::floor, Expr> floor(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::round, Expr> round(const Expression<Expr> &u);
template<typename Expr>
const UnaryOpExpression<KernelOp::rint, Expr> rint(const Expression<Expr> &u);

template<typename Expr, typename T, typename>
const BinaryOpExpression<KernelOp::scalarExpPow<T>, Expr, T> pow(const Expression<Expr> &base, const T &exponent);
template<typename Expr, typename T, typename>
const BinaryOpExpression<KernelOp::scalarExpPowf<T>, Expr, T> powf(const Expression<Expr> &base, const T &exponent);
template<typename Expr, typename T, typename>
const BinaryOpExpression<KernelOp::scalarMaxRightf<T>, Expr, T> maxf(const Expression<Expr> &lhs, const T &n);
template<typename Expr, typename T, typename>
const BinaryOpExpression<KernelOp::scalarMinRightf<T>, Expr, T> minf(const Expression<Expr> &lhs, const T &n);

}

#endif
