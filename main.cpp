#include "inc/libcumat.h"
#include <iostream>
#include <time.h>

int main(int argc, char const* argv[])
{
	srand(time(0));
	cumat<double> mat(4, 5);
	mat.rand();
	std::cout << mat << std::endl << std::endl;
	mat = mat + 2;
	std::cout << mat << std::endl << std::endl;
	mat = mat.transpose();
	std::cout << mat << std::endl;
	return 0;
}
