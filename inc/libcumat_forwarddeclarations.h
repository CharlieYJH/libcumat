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

}

#endif
