#ifndef LIBCUMAT_UNARYOP_H_
#define LIBCUMAT_UNARYOP_H_

#include <vector>

namespace Cumat
{
namespace KernelOp
{

class UnaryOp
{
	const std::string op_;
	
	public:
	UnaryOp(const std::string &op) : op_(op) {}
	template<typename Expr>
	std::string operator ()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string var = u.buildKernel(params, num, args, transpose, has_transpose_expr);
		return op_ + "(" + var + ")";
	}
};

struct negative : public UnaryOp
{
	negative(void) : UnaryOp("-") {}
};

struct abs : public UnaryOp
{
	abs(void) : UnaryOp("abs") {}
};

struct exp : public UnaryOp
{
	exp(void) : UnaryOp("exp") {}
};

struct exp10 : public UnaryOp
{
	exp10(void) : UnaryOp("exp10") {}
};

struct exp2 : public UnaryOp
{
	exp2(void) : UnaryOp("exp2") {}
};

struct log : public UnaryOp
{
	log(void) : UnaryOp("log") {}
};

struct log1p : public UnaryOp
{
	log1p(void) : UnaryOp("log1p") {}
};

struct log10 : public UnaryOp
{
	log10(void) : UnaryOp("log10") {}
};

struct log2 : public UnaryOp
{
	log2(void) : UnaryOp("log2") {}
};

struct square
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string var = u.buildKernel(params, num, args, transpose, has_transpose_expr);
		return "(" + var + "*" + var + ")";
	}
};

struct sqrt : public UnaryOp
{
	sqrt(void) : UnaryOp("sqrt") {}
};

struct rsqrt : public UnaryOp
{
	rsqrt(void) : UnaryOp("rsqrt") {}
};

struct cube
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string var = u.buildKernel(params, num, args, transpose, has_transpose_expr);
		return "(" + var + "*" + var + "*" + var + ")";
	}
};

struct cbrt : public UnaryOp
{
	cbrt(void) : UnaryOp("cbrt") {}
};

struct rcbrt : public UnaryOp
{
	rcbrt(void) : UnaryOp("rcbrt") {}
};

struct sin : public UnaryOp
{
	sin(void) : UnaryOp("sin") {}
};

struct asin : public UnaryOp
{
	asin(void) : UnaryOp("asin") {}
};

struct sinh : public UnaryOp
{
	sinh(void) : UnaryOp("sinh") {}
};

struct asinh : public UnaryOp
{
	asinh(void) : UnaryOp("asinh") {}
};

struct cos : public UnaryOp
{
	cos(void) : UnaryOp("cos") {}
};

struct acos : public UnaryOp
{
	acos(void) : UnaryOp("acos") {}
};

struct cosh : public UnaryOp
{
	cosh(void) : UnaryOp("cosh") {}
};

struct acosh : public UnaryOp
{
	acosh(void) : UnaryOp("acosh") {}
};

struct tan : public UnaryOp
{
	tan(void) : UnaryOp("tan") {}
};

struct atan : public UnaryOp
{
	atan(void) : UnaryOp("atan") {}
};

struct tanh : public UnaryOp
{
	tanh(void) : UnaryOp("tanh") {}
};

struct atanh : public UnaryOp
{
	atanh(void) : UnaryOp("atanh") {}
};

struct ceil : public UnaryOp
{
	ceil(void) : UnaryOp("ceil") {}
};

struct floor : public UnaryOp
{
	floor(void) : UnaryOp("floor") {}
};

struct round : public UnaryOp
{
	round(void) : UnaryOp("rint") {}
};

}

}

#endif
