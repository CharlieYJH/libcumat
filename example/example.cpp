#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::Matrixf mat(4, 5);
	mat.rand();
	
	std::cout << mat << std::endl;

    mat.sigmoid();

    std::cout << mat << std::endl;

	return 0;
}
