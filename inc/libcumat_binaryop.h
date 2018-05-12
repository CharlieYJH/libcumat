#ifndef LIBCUMAT_BINARYOP_H_
#define LIBCUMAT_BINARYOP_H_

#include <vector>

#include "libcumat_typestring.h"

namespace Cumat
{
namespace KernelOp
{

class BinaryVectorOp
{
	const std::string preop_;
	const std::string midop_;
	const std::string cast_;
	
	public:
	BinaryVectorOp(const std::string &preop, const std::string &midop) : preop_(preop), midop_(" " + midop + " "), cast_("") {}
	BinaryVectorOp(const std::string &preop, const std::string &midop, const std::string &cast) : preop_(preop), midop_(" " + midop + " "), cast_("(" + cast + ")") {}
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string lhs = u.buildKernel(params, num, args, transpose, has_transpose_expr);
		std::string rhs = v.buildKernel(params, num, args, transpose, has_transpose_expr);
		return preop_ + "(" + cast_ + lhs + midop_ + cast_ + rhs + ")";
	}
};

template<typename T>
class BinaryScalarOp
{
	protected:
	const std::string preop_;
	const std::string midop_;
	const std::string cast_;

	public:
	BinaryScalarOp(const std::string &preop, const std::string &midop, const std::string &cast) : preop_(preop), midop_(" " + midop + " "), cast_(cast) {}
};

template<typename T>
class BinaryScalarOpRight : public BinaryScalarOp<T>
{
	public:
	BinaryScalarOpRight(const std::string &preop, const std::string &midop) : BinaryScalarOp(preop, midop, "") {}
	BinaryScalarOpRight(const std::string &preop, const std::string &midop, const std::string &cast) : BinaryScalarOp(preop, midop, "(" + cast + ")") {}
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string lhs = u.buildKernel(params, num, args, transpose, has_transpose_expr);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + TypeString<T>::type + " s" + id_num;
		args.push_back((void *)&n);

		return preop_ + "(" + cast_ + lhs + midop_ + cast_ + rhs + ")";
	}
};

template<typename T>
class BinaryScalarOpLeft : public BinaryScalarOp<T>
{
	public:
	BinaryScalarOpLeft(const std::string &preop, const std::string &midop) : BinaryScalarOp(preop, midop, "") {}
	BinaryScalarOpLeft(const std::string &preop, const std::string &midop, const std::string &cast) : BinaryScalarOp(preop, midop, "(" + cast + ")") {}
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args, const bool &transpose, bool &has_transpose_expr) const
	{
		std::string id_num = std::to_string(num++);
		std::string lhs = "s" + id_num;
		params += ", " + TypeString<T>::type + " s" + id_num;
		args.push_back((void *)&n);

		std::string rhs = u.buildKernel(params, num, args, transpose, has_transpose_expr);

		return preop_ + "(" + cast_ + lhs + midop_ + cast_ + rhs + ")";
	}
};

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
	vectorPow(void) : BinaryVectorOp("pow", ",", "double") {}
};

struct vectorPowf : public BinaryVectorOp
{
	vectorPowf(void) : BinaryVectorOp("powf", ",", "float") {}
};

template<typename T>
struct scalarExpPow : public BinaryScalarOpRight<T>
{
	scalarExpPow(void) : BinaryScalarOpRight("pow", ",", "double") {}
};

template<typename T>
struct scalarExpPowf : public BinaryScalarOpRight<T>
{
	scalarExpPowf(void) : BinaryScalarOpRight("powf", ",", "float") {}
};

template<typename T>
struct scalarBasePow : public BinaryScalarOpLeft<T>
{
	scalarBasePow(void) : BinaryScalarOpLeft("pow", ",", "double") {}
};

template<typename T>
struct scalarBasePowf : public BinaryScalarOpLeft<T>
{
	scalarBasePowf(void) : BinaryScalarOpLeft("powf", ",", "float") {}
};

struct vectorAtan2 : public BinaryVectorOp
{
	vectorAtan2(void) : BinaryVectorOp("atan2", ",", "double") {}
};

struct vectorAtan2f : public BinaryVectorOp
{
	vectorAtan2f(void) : BinaryVectorOp("atan2f", ",", "float") {}
};

template<typename T>
struct scalarAtan2Right : public BinaryScalarOpRight<T>
{
	scalarAtan2Right(void) : BinaryScalarOpRight("atan2", ",", "double") {}
};

template<typename T>
struct scalarAtan2Rightf : public BinaryScalarOpRight<T>
{
	scalarAtan2Rightf(void) : BinaryScalarOpRight("atan2f", ",", "float") {}
};

template<typename T>
struct scalarAtan2Left : public BinaryScalarOpLeft<T>
{
	scalarAtan2Left(void) : BinaryScalarOpLeft("atan2", ",", "double") {}
};

template<typename T>
struct scalarAtan2Leftf : public BinaryScalarOpLeft<T>
{
	scalarAtan2Leftf(void) : BinaryScalarOpLeft("atan2f", ",", "float") {}
};

struct vectorMax : public BinaryVectorOp
{
	vectorMax(void) : BinaryVectorOp("fmax", ",", "double") {}
};

struct vectorMaxf : public BinaryVectorOp
{
	vectorMaxf(void) : BinaryVectorOp("fmaxf", ",", "float") {}
};

template<typename T>
struct scalarMaxRight : public BinaryScalarOpRight<T>
{
	scalarMaxRight(void) : BinaryScalarOpRight("fmax", ",", "double") {}
};

template<typename T>
struct scalarMaxRightf : public BinaryScalarOpRight<T>
{
	scalarMaxRightf(void) : BinaryScalarOpRight("fmaxf", ",", "float") {}
};

template<typename T>
struct scalarMaxLeft : public BinaryScalarOpLeft<T>
{
	scalarMaxLeft(void) : BinaryScalarOpLeft("fmax", ",", "double") {}
};

template<typename T>
struct scalarMaxLeftf : public BinaryScalarOpLeft<T>
{
	scalarMaxLeftf(void) : BinaryScalarOpLeft("fmaxf", ",", "float") {}
};

struct vectorMin : public BinaryVectorOp
{
	vectorMin(void) : BinaryVectorOp("fmin", ",", "double") {}
};

struct vectorMinf : public BinaryVectorOp
{
	vectorMinf(void) : BinaryVectorOp("fminf", ",", "float") {}
};

template<typename T>
struct scalarMinRight : public BinaryScalarOpRight<T>
{
	scalarMinRight(void) : BinaryScalarOpRight("fmin", ",", "double") {}
};

template<typename T>
struct scalarMinRightf : public BinaryScalarOpRight<T>
{
	scalarMinRightf(void) : BinaryScalarOpRight("fminf", ",", "float") {}
};

template<typename T>
struct scalarMinLeft : public BinaryScalarOpLeft<T>
{
	scalarMinLeft(void) : BinaryScalarOpLeft("fmin", ",", "double") {}
};

template<typename T>
struct scalarMinLeftf : public BinaryScalarOpLeft<T>
{
	scalarMinLeftf(void) : BinaryScalarOpLeft("fminf", ",", "float") {}
};

}
}

#endif
