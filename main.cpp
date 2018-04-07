#include <iostream>
#include <chrono>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::createCublasHandle();
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(5, 3)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(3, 6)));
	Cumat::Matrixd tmp(5, 6);

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	std::cout << "tmp = " << std::endl << mat.minElement() << std::endl << std::endl;
	std::cout << "tmp = " << std::endl << mat.minIndex() << std::endl << std::endl;

	Cumat::Matrixd mat3(2000, 2000);
	Cumat::Matrixd tmp2(2000, 2001);
	Cumat::Matrixd tmp3(2000, 2001);

	Cumat::Matrixd Wy(Cumat::Matrixd::random(512, 512));
	Cumat::Matrixd dy(Cumat::Matrixd::random(512, 512));
	Cumat::Matrixd tmp4(Wy.rows(), dy.cols());

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++) {
		Wy.mmul(dy, tmp4);
		Wy.mmul(dy, tmp4);
		Wy.mmul(dy, tmp4);
		Wy.mmul(dy, tmp4);
	}
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	// std::cout << Wy << std::endl;

	Cumat::destroyCublasHandle();

	return 0;
}
