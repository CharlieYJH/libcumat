#include "inc/libcumat.h"
#include <iostream>

int main(int argc, char const* argv[])
{
	Cumat<double> mat = std::move(Cumat<double>::random(5, 3));
	Cumat<double> mat2 = std::move(Cumat<double>::random(3, 6));
	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;
	mat = ~mat2 ^ ~mat;
	std::cout << "A x B = " << std::endl << mat << std::endl;
	return 0;
}
