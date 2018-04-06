#include "inc/libcumat.h"
#include <iostream>
#include <chrono>

int main(int argc, char const* argv[])
{
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(5, 3)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(3, 6)));

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	mat = ~mat2 ^ ~mat;

	std::cout << "A x B = " << std::endl << mat << std::endl << std::endl;
	
	mat.isigmoid();

	std::cout << "Op(A x B) = " << std::endl << mat << std::endl << std::endl;

	Cumat::Matrixd mat3(4000, 4000);
	mat3.rand();
	auto start = std::chrono::high_resolution_clock::now();
	mat3 += mat3;
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	// Cumat::Matrixd mat3(std::move(Cumat::Matrixd::random(512, 512)));
	// Cumat::Matrixd mat4(512, 512);
	// Cumat::Matrixd mat5(std::move(Cumat::Matrixd::random(4 * 512, 512)));

	// auto start = std::chrono::high_resolution_clock::now();
	// mat4 = ((mat5 ^ mat3) + (mat5 ^ mat3)).tanh();
	// auto end = std::chrono::high_resolution_clock::now();

	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// std::cout << duration << std::endl;

	return 0;
}
