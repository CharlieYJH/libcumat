#ifndef LIBCUMAT_ADDITION_H_
#define LIBCUMAT_ADDITION_H_

#include <vector>

namespace Cumat
{
namespace KernelOp
{

class BinaryVectorOp
{
	const std::string preop_;
	const std::string midop_;
	
	public:
	BinaryVectorOp(const std::string &preop, const std::string &midop) : preop_(preop), midop_(midop) {}
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string lhs = u.buildKernel(params, num, args, transpose);
		std::string rhs = v.buildKernel(params, num, args, transpose);
		return preop_ + "(" + lhs + midop_ + rhs + ")";
	}
};

template<typename T>
class BinaryScalarOpRight
{
	const std::string preop_;
	const std::string midop_;
	static const std::string type_;

	public:
	BinaryScalarOpRight(const std::string &preop, const std::string &midop) : preop_(preop), midop_(midop) {}
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string lhs = u.buildKernel(params, num, args, transpose);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return preop_ + "(" + lhs + midop_ + rhs + ")";
	}
};
template<> const std::string BinaryScalarOpRight<double>::type_ = "double";
template<> const std::string BinaryScalarOpRight<float>::type_ = "float";

template<typename T>
class BinaryScalarOpLeft
{
	const std::string preop_;
	const std::string midop_;
	static const std::string type_;

	public:
	BinaryScalarOpLeft(const std::string &preop, const std::string &midop) : preop_(preop), midop_(midop) {}
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string id_num = std::to_string(num++);
		std::string lhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		std::string rhs = u.buildKernel(params, num, args, transpose);

		return preop_ + "(" + lhs + midop_ + rhs + ")";
	}
};
template<> const std::string BinaryScalarOpLeft<double>::type_ = "double";
template<> const std::string BinaryScalarOpLeft<float>::type_ = "float";

struct vectorSum : public BinaryVectorOp
{
	vectorSum(void) : BinaryVectorOp("", "+") {}
};

template<typename T>
struct scalarSum : public BinaryScalarOpRight<T>
{
	scalarSum(void) : BinaryScalarOpRight("", "+") {}
};

struct vectorSub : public BinaryVectorOp
{
	vectorSub(void) : BinaryVectorOp("", "-") {}
};

template<typename T>
struct scalarSubRight : public BinaryScalarOpRight<T>
{
	scalarSubRight(void) : BinaryScalarOpRight("", "-") {}
};

template<typename T>
struct scalarSubLeft : public BinaryScalarOpLeft<T>
{
	scalarSubLeft(void) : BinaryScalarOpLeft("", "-") {}
};

struct vectorMul : public BinaryVectorOp
{
	vectorMul(void) : BinaryVectorOp("", "*") {}
};

template<typename T>
struct scalarMul : public BinaryScalarOpRight<T>
{
	scalarMul(void) : BinaryScalarOpRight("", "*") {}
};

struct vectorDiv : public BinaryVectorOp
{
	vectorDiv(void) : BinaryVectorOp("", "/") {}
};

template<typename T>
struct scalarDivRight : public BinaryScalarOpRight<T>
{
	scalarDivRight(void) : BinaryScalarOpRight("", "/") {}
};

template<typename T>
struct scalarDivLeft : public BinaryScalarOpLeft<T>
{
	scalarDivLeft(void) : BinaryScalarOpLeft("", "/") {}
};

struct vectorPow : public BinaryVectorOp
{
	vectorPow(void) : BinaryVectorOp("pow", ",") {}
};

template<typename T>
struct scalarExpPow : public BinaryScalarOpRight<T>
{
	scalarExpPow(void) : BinaryScalarOpRight("pow", ",") {}
};

template<typename T>
struct scalarBasePow : public BinaryScalarOpLeft<T>
{
	scalarBasePow(void) : BinaryScalarOpLeft("pow", ",") {}
};

struct vectorAtan2 : public BinaryVectorOp
{
	vectorAtan2(void) : BinaryVectorOp("atan2", ",") {}
};

template<typename T>
struct scalarAtan2Right : public BinaryScalarOpRight<T>
{
	scalarAtan2Right(void) : BinaryScalarOpRight("atan2", ",") {}
};

template<typename T>
struct scalarAtan2Left : public BinaryScalarOpLeft<T>
{
	scalarAtan2Left(void) : BinaryScalarOpLeft("atan2", ",") {}
};

class UnaryOp
{
	const std::string op_;
	
	public:
	UnaryOp(const std::string &op) : op_(op) {}
	template<typename Expr>
	std::string operator ()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string var = u.buildKernel(params, num, args, transpose);
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
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string var = u.buildKernel(params, num, args, transpose);
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
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args, const bool &transpose) const
	{
		std::string var = u.buildKernel(params, num, args, transpose);
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
	round(void) : UnaryOp("round") {}
};

}
}

#endif
