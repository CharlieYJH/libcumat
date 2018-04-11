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

struct negative
{
	template<typename Expr>
	std::string operator()(const Expr &u, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string var = u.buildKernel(params, num, args);
		return "-(" + var + ")";
	}
};

}
}

#endif
