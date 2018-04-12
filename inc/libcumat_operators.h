#ifndef LIBCUMAT_ADDITION_H_
#define LIBCUMAT_ADDITION_H_

#include <vector>

namespace Cumat
{
namespace KernelOp
{

struct vectorSum
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);
		std::string rhs = v.buildKernel(params, num, args);
		return "(" + lhs + "+" + rhs + ")";
	}
};

template<typename T>
struct scalarSum
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "+" + rhs + ")";
	}
};
template<> const std::string scalarSum<double>::type_ = "double";
template<> const std::string scalarSum<float>::type_ = "float";

struct vectorSub
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);
		std::string rhs = v.buildKernel(params, num, args);
		return "(" + lhs + "-" + rhs + ")";
	}
};

template<typename T>
struct scalarSubRight
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "-" + rhs + ")";
	}
};
template<> const std::string scalarSubRight<double>::type_ = "double";
template<> const std::string scalarSubRight<float>::type_ = "float";

template<typename T>
struct scalarSubLeft
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string id_num = std::to_string(num++);
		std::string lhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		std::string rhs = u.buildKernel(params, num, args);

		return "(" + lhs + "-" + rhs + ")";
	}
};
template<> const std::string scalarSubLeft<double>::type_ = "double";
template<> const std::string scalarSubLeft<float>::type_ = "float";

struct vectorMul
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);
		std::string rhs = v.buildKernel(params, num, args);
		return "(" + lhs + "*" + rhs + ")";
	}
};

template<typename T>
struct scalarMul
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "*" + rhs + ")";
	}
};
template<> const std::string scalarMul<double>::type_ = "double";
template<> const std::string scalarMul<float>::type_ = "float";

struct vectorDiv
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);
		std::string rhs = v.buildKernel(params, num, args);
		return "(" + lhs + "/" + rhs + ")";
	}
};

template<typename T>
struct scalarDivRight
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "/" + rhs + ")";
	}
};
template<> const std::string scalarDivRight<double>::type_ = "double";
template<> const std::string scalarDivRight<float>::type_ = "float";

template<typename T>
struct scalarDivLeft
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string id_num = std::to_string(num++);
		std::string lhs = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		std::string rhs = u.buildKernel(params, num, args);

		return "(" + lhs + "/" + rhs + ")";
	}
};
template<> const std::string scalarDivLeft<double>::type_ = "double";
template<> const std::string scalarDivLeft<float>::type_ = "float";

struct vectorPow
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string base = u.buildKernel(params, num, args);
		std::string exponent = v.buildKernel(params, num, args);
		return "pow((double)(" + base + "),(double)(" + exponent + "))";
	}
};

template<typename T>
struct scalarExpPow
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string base = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string exponent = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "pow((double)(" + base + "),(double)(" + exponent + "))";
	}
};
template<> const std::string scalarExpPow<double>::type_ = "double";
template<> const std::string scalarExpPow<float>::type_ = "float";

template<typename T>
struct scalarBasePow
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string id_num = std::to_string(num++);
		std::string base = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		std::string exponent = u.buildKernel(params, num, args);

		return "pow((double)(" + base + "),(double)(" + exponent + "))";
	}
};
template<> const std::string scalarBasePow<double>::type_ = "double";
template<> const std::string scalarBasePow<float>::type_ = "float";

struct vectorAtan2
{
	template<typename Expr1, typename Expr2>
	std::string operator()(const Expr1 &u, const Expr2 &v, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string y = u.buildKernel(params, num, args);
		std::string x = v.buildKernel(params, num, args);
		return "atan2(" + y + "," + x + ")";
	}
};

template<typename T>
struct scalarAtan2Right
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string y = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string x = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		return "atan2(" + y + "," + x + ")";
	}
};
template<> const std::string scalarAtan2Right<double>::type_ = "double";
template<> const std::string scalarAtan2Right<float>::type_ = "float";

template<typename T>
struct scalarAtan2Left
{
	static const std::string type_;
	template<typename Expr>
	std::string operator()(const Expr &u, const T &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string id_num = std::to_string(num++);
		std::string y = "s" + id_num;
		params += ", " + type_ + " s" + id_num;
		args.push_back((void *)&n);

		std::string x = u.buildKernel(params, num, args);

		return "atan2(" + y + "," + x + ")";
	}
};
template<> const std::string scalarAtan2Left<double>::type_ = "double";
template<> const std::string scalarAtan2Left<float>::type_ = "float";

struct negative
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "-(" + var + ")";
	}
};

struct abs
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "abs(" + var + ")";
	}
};

struct exp
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "exp(" + var + ")";
	}
};

struct exp10
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "exp10(" + var + ")";
	}
};

struct exp2
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "exp2(" + var + ")";
	}
};

struct log
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "log(" + var + ")";
	}
};

struct log1p
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "log1p(" + var + ")";
	}
};

struct log10
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "log10(" + var + ")";
	}
};

struct log2
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "log2(" + var + ")";
	}
};

struct square
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "(" + var + "*" + var + ")";
	}
};

struct sqrt
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "sqrt(" + var + ")";
	}
};

struct rsqrt
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "rsqrt(" + var + ")";
	}
};

struct cube
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "(" + var + "*" + var + "*" + var + ")";
	}
};

struct cbrt
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "cbrt(" + var + ")";
	}
};

struct rcbrt
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "rcbrt(" + var + ")";
	}
};

struct sin
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "sin(" + var + ")";
	}
};

struct asin
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "asin(" + var + ")";
	}
};

struct sinh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "sinh(" + var + ")";
	}
};

struct asinh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "asinh(" + var + ")";
	}
};

struct cos
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "cos(" + var + ")";
	}
};

struct acos
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "acos(" + var + ")";
	}
};

struct cosh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "cosh(" + var + ")";
	}
};

struct acosh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "acosh(" + var + ")";
	}
};

struct tan
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "tan(" + var + ")";
	}
};

struct atan
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "atan(" + var + ")";
	}
};

struct tanh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "tanh(" + var + ")";
	}
};

struct atanh
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "atanh(" + var + ")";
	}
};

struct ceil
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "ceil(" + var + ")";
	}
};

struct floor
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "floor(" + var + ")";
	}
};

struct round
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "round(" + var + ")";
	}
};

}
}

#endif
