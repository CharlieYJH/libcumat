#ifndef LIBCUMAT_BINARYOPEXPRESSION_H_
#define LIBCUMAT_BINARYOPEXPRESSION_H_

#include <vector>
#include <type_traits>

#include "libcumat_expression.h"
#include "libcumat_binaryop.h"

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

    std::string buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const;
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
std::string BinaryOpExpression<Op, Expr1, Expr2>::buildKernel(std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
{
    return op_(u_, v_, params, num, args, transpose, has_transpose_expr);
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

// Compile time check for whether type T is an integer type (prevents ambiguous overloads)
template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T> operator+(const Expression<Expr> &lhs, const T &n)
{
    return BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T>(lhs, n);
}

// Explicit templates for floating point argument overloads allows references to be implicitly cast into
// a floating point type, which makes writing matrix expressions involving references more natural
template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float> operator+(const Expression<Expr> &lhs, const float &n)
{
    return BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float>(lhs, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double> operator+(const Expression<Expr> &lhs, const double &n)
{
    return BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double>(lhs, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<long double>, Expr, long double> operator+(const Expression<Expr> &lhs, const long double &n)
{
    return BinaryOpExpression<KernelOp::scalarSum<long double>, Expr, long double>(lhs, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T> operator+(const T &n, const Expression<Expr> &rhs)
{
    return BinaryOpExpression<KernelOp::scalarSum<T>, Expr, T>(rhs, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float> operator+(const float &n, const Expression<Expr> &rhs)
{
    return BinaryOpExpression<KernelOp::scalarSum<float>, Expr, float>(rhs, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double> operator+(const double &n, const Expression<Expr> &rhs)
{
    return BinaryOpExpression<KernelOp::scalarSum<double>, Expr, double>(rhs, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSum<long double>, Expr, long double> operator+(const long double &n, const Expression<Expr> &rhs)
{
    return BinaryOpExpression<KernelOp::scalarSum<long double>, Expr, long double>(rhs, n);
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

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSubRight<T>, Expr, T> operator-(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarSubRight<T>, Expr, T>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubRight<float>, Expr, float> operator-(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarSubRight<float>, Expr, float>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubRight<double>, Expr, double> operator-(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarSubRight<double>, Expr, double>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubRight<long double>, Expr, long double> operator-(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarSubRight<long double>, Expr, long double>(u, n); 
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarSubLeft<T>, Expr, T> operator-(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarSubLeft<T>, Expr, T>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubLeft<float>, Expr, float> operator-(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarSubLeft<float>, Expr, float>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubLeft<double>, Expr, double> operator-(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarSubLeft<double>, Expr, double>(u, n); 
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarSubLeft<long double>, Expr, long double> operator-(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarSubLeft<long double>, Expr, long double>(u, n); 
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

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T> operator*(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float> operator*(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double> operator*(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<long double>, Expr, long double> operator*(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMul<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T> operator*(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMul<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float> operator*(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMul<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double> operator*(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMul<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMul<long double>, Expr, long double> operator*(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMul<long double>, Expr, long double>(u, n);
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

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarDivRight<T>, Expr, T> operator/(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarDivRight<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivRight<float>, Expr, float> operator/(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarDivRight<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivRight<double>, Expr, double> operator/(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarDivRight<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivRight<long double>, Expr, long double> operator/(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarDivRight<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarDivLeft<T>, Expr, T> operator/(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarDivLeft<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivLeft<float>, Expr, float> operator/(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarDivLeft<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivLeft<double>, Expr, double> operator/(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarDivLeft<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarDivLeft<long double>, Expr, long double> operator/(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarDivLeft<long double>, Expr, long double>(u, n);
}

// -------------- Pow --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorPow, Expr1, Expr2> pow(const Expression<Expr1> &base, const Expression<Expr2> &exponent)
{
    const Expr1 &u = base;
    const Expr2 &v = exponent;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorPow, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarExpPow<T>, Expr, T> pow(const Expression<Expr> &base, const T &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPow<T>, Expr, T>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPow<float>, Expr, float> pow(const Expression<Expr> &base, const float &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPow<float>, Expr, float>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPow<double>, Expr, double> pow(const Expression<Expr> &base, const double &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPow<double>, Expr, double>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPow<long double>, Expr, long double> pow(const Expression<Expr> &base, const long double &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPow<long double>, Expr, long double>(u, exponent);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarBasePow<T>, Expr, T> pow(const T &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePow<T>, Expr, T>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePow<float>, Expr, float> pow(const float &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePow<float>, Expr, float>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePow<double>, Expr, double> pow(const double &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePow<double>, Expr, double>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePow<long double>, Expr, long double> pow(const long double &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePow<long double>, Expr, long double>(u, base);
}

// -------------- Powf --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorPowf, Expr1, Expr2> powf(const Expression<Expr1> &base, const Expression<Expr2> &exponent)
{
    const Expr1 &u = base;
    const Expr2 &v = exponent;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorPowf, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarExpPowf<T>, Expr, T> powf(const Expression<Expr> &base, const T &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPowf<T>, Expr, T>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPowf<float>, Expr, float> powf(const Expression<Expr> &base, const float &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPowf<float>, Expr, float>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPowf<double>, Expr, double> powf(const Expression<Expr> &base, const double &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPowf<double>, Expr, double>(u, exponent);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarExpPowf<long double>, Expr, long double> powf(const Expression<Expr> &base, const long double &exponent)
{
    const Expr &u = base;
    return BinaryOpExpression<KernelOp::scalarExpPowf<long double>, Expr, long double>(u, exponent);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarBasePowf<T>, Expr, T> powf(const T &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePowf<T>, Expr, T>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePowf<float>, Expr, float> powf(const float &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePowf<float>, Expr, float>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePowf<double>, Expr, double> powf(const double &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePowf<double>, Expr, double>(u, base);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarBasePowf<long double>, Expr, long double> powf(const long double &base, const Expression<Expr> &exponent)
{
    const Expr &u = exponent;
    return BinaryOpExpression<KernelOp::scalarBasePowf<long double>, Expr, long double>(u, base);
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

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Right<T>, Expr, T> atan2(const Expression<Expr> &y, const T &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Right<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Right<float>, Expr, float> atan2(const Expression<Expr> &y, const float &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Right<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Right<double>, Expr, double> atan2(const Expression<Expr> &y, const double &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Right<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Right<long double>, Expr, long double> atan2(const Expression<Expr> &y, const long double &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Right<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Left<T>, Expr, T> atan2(const T &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Left<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Left<float>, Expr, float> atan2(const float &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Left<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Left<double>, Expr, double> atan2(const double &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Left<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Left<long double>, Expr, long double> atan2(const long double &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Left<long double>, Expr, long double>(u, n);
}

// -------------- Atan2f --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorAtan2f, Expr1, Expr2> atan2f(const Expression<Expr1> &y, const Expression<Expr2> &x)
{
    const Expr1 &u = y;
    const Expr2 &v = x;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorAtan2f, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Rightf<T>, Expr, T> atan2f(const Expression<Expr> &y, const T &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Rightf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Rightf<float>, Expr, float> atan2f(const Expression<Expr> &y, const float &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Rightf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Rightf<double>, Expr, double> atan2f(const Expression<Expr> &y, const double &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Rightf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Rightf<long double>, Expr, long double> atan2f(const Expression<Expr> &y, const long double &n)
{
    const Expr &u = y;
    return BinaryOpExpression<KernelOp::scalarAtan2Rightf<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarAtan2Leftf<T>, Expr, T> atan2f(const T &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Leftf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Leftf<float>, Expr, float> atan2f(const float &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Leftf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Leftf<double>, Expr, double> atan2f(const double &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Leftf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarAtan2Leftf<long double>, Expr, long double> atan2f(const long double &n, const Expression<Expr> &x)
{
    const Expr &u = x;
    return BinaryOpExpression<KernelOp::scalarAtan2Leftf<long double>, Expr, long double>(u, n);
}

// -------------- Max --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorMax, Expr1, Expr2> max(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
    const Expr1 &u = lhs;
    const Expr2 &v = rhs;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorMax, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMaxRight<T>, Expr, T> max(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRight<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRight<float>, Expr, float> max(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRight<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRight<double>, Expr, double> max(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRight<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRight<long double>, Expr, long double> max(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRight<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMaxLeft<T>, Expr, T> max(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeft<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeft<float>, Expr, float> max(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeft<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeft<double>, Expr, double> max(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeft<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeft<long double>, Expr, long double> max(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeft<long double>, Expr, long double>(u, n);
}

// -------------- Maxf --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorMaxf, Expr1, Expr2> maxf(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
    const Expr1 &u = lhs;
    const Expr2 &v = rhs;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorMaxf, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMaxRightf<T>, Expr, T> maxf(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRightf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRightf<float>, Expr, float> maxf(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRightf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRightf<double>, Expr, double> maxf(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRightf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxRightf<long double>, Expr, long double> maxf(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMaxRightf<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMaxLeftf<T>, Expr, T> maxf(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeftf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeftf<float>, Expr, float> maxf(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeftf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeftf<double>, Expr, double> maxf(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeftf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMaxLeftf<long double>, Expr, long double> maxf(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMaxLeftf<long double>, Expr, long double>(u, n);
}

// -------------- Min --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorMin, Expr1, Expr2> min(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
    const Expr1 &u = lhs;
    const Expr2 &v = rhs;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorMin, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMinRight<T>, Expr, T> min(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRight<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRight<float>, Expr, float> min(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRight<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRight<double>, Expr, double> min(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRight<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRight<long double>, Expr, long double> min(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRight<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMinLeft<T>, Expr, T> min(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeft<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeft<float>, Expr, float> min(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeft<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeft<double>, Expr, double> min(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeft<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeft<long double>, Expr, long double> min(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeft<long double>, Expr, long double>(u, n);
}

// -------------- Minf --------------

template<typename Expr1, typename Expr2>
const BinaryOpExpression<KernelOp::vectorMinf, Expr1, Expr2> minf(const Expression<Expr1> &lhs, const Expression<Expr2> &rhs)
{
    const Expr1 &u = lhs;
    const Expr2 &v = rhs;
    assert(u.rows() == v.rows() && u.cols() == v.cols());
    return BinaryOpExpression<KernelOp::vectorMinf, Expr1, Expr2>(u, v);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMinRightf<T>, Expr, T> minf(const Expression<Expr> &lhs, const T &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRightf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRightf<float>, Expr, float> minf(const Expression<Expr> &lhs, const float &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRightf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRightf<double>, Expr, double> minf(const Expression<Expr> &lhs, const double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRightf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinRightf<long double>, Expr, long double> minf(const Expression<Expr> &lhs, const long double &n)
{
    const Expr &u = lhs;
    return BinaryOpExpression<KernelOp::scalarMinRightf<long double>, Expr, long double>(u, n);
}

template<typename Expr, typename T, typename = typename std::enable_if<std::is_integral<T>::value, void>::type>
const BinaryOpExpression<KernelOp::scalarMinLeftf<T>, Expr, T> minf(const T &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeftf<T>, Expr, T>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeftf<float>, Expr, float> minf(const float &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeftf<float>, Expr, float>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeftf<double>, Expr, double> minf(const double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeftf<double>, Expr, double>(u, n);
}

template<typename Expr>
const BinaryOpExpression<KernelOp::scalarMinLeftf<long double>, Expr, long double> minf(const long double &n, const Expression<Expr> &rhs)
{
    const Expr &u = rhs;
    return BinaryOpExpression<KernelOp::scalarMinLeftf<long double>, Expr, long double>(u, n);
}

}

#endif
