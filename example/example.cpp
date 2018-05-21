#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::Matrixf mat(4, 5);
	mat.rand();
	
	std::cout << mat.rows() << " " << mat.cols() << std::endl;
	std::cout << mat << std::endl;

	return 0;
}
