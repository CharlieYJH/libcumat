#include "inc/libcumat.h"
#include <iostream>

int main(int argc, char const* argv[])
{
	Cumat<double> mat(5, 1);
	Cumat<double> mat2(1, 6);
	mat.rand();
	mat2.rand();
	std::cout << mat << std::endl << std::endl;
	std::cout << mat2 << std::endl << std::endl;
	mat = mat - 3;
	std::cout << mat << std::endl << std::endl;
	mat = mat.mmul(mat2);
	std::cout << mat << std::endl;
	return 0;
}
