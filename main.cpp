#include "inc/libcumat.h"
#include <iostream>

int main(int argc, char const* argv[])
{
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(5, 3)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(3, 6)));

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	mat = ~mat2 ^ ~mat;

	std::cout << "A x B = " << std::endl << mat << std::endl << std::endl;
	std::cout << mat.tanh() << std::endl << std::endl;

	return 0;
}
