#ifndef LIBCUMAT_ADDITION_H_
#define LIBCUMAT_ADDITION_H_

#include <vector>

namespace Cumat
{
namespace kernelOp
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

struct scalarSum
{
	template<typename Expr>
	std::string operator()(const Expr &u, const double &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", double s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "+" + rhs + ")";
	}

	template<typename Expr>
	std::string operator()(const Expr &u, const float &n, std::string &params, int &num, std::vector<void *> &args) const
	{
		std::string lhs = u.buildKernel(params, num, args);

		std::string id_num = std::to_string(num++);
		std::string rhs = "s" + id_num;
		params += ", float s" + id_num;
		args.push_back((void *)&n);

		return "(" + lhs + "+" + rhs + ")";
	}
};

}
}

#endif
