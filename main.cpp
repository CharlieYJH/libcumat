#include <iostream>
#include <chrono>
#include <typeinfo>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixf mat(std::move(Cumat::Matrixf::random(5, 6)));
	Cumat::Matrixf mat2(std::move(Cumat::Matrixf::random(5, 6)));
	Cumat::Matrixf result(5, 6);

	mat = mat + mat * mat + 1.0f;
	cudaDeviceSynchronize();

	mat.fill(1);
	std::cout << (abs(mat - 10 * mat) - mat / 4).eval<float>() << std::endl;

	// std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	// std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	mat = mat + mat * mat + 1.0f;
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	// std::cout << "Result = " << std::endl << mat << std::endl << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
