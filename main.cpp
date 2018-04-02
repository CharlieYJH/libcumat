#include "inc/libcumat.h"
#include <iostream>
#include <time.h>

int main(int argc, char const* argv[])
{
	srand(time(0));
	cumat<double> mat(5, 5);
	mat.fill(2);
	std::cout << mat << std::endl;
	mat.rand();
	std::cout << mat << std::endl;
	mat.rand(-5, 7);
	std::cout << mat << std::endl;
	return 0;
}
