#include <iostream>
#include <chrono>
#include <typeinfo>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::createCublasHandle();
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(5, 3)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(3, 6)));
	Cumat::Matrixd tmp(512, 512);
	Cumat::Matrixd tmp5(512, 512);

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	std::cout << "tmp = " << std::endl << mat.minElement() << std::endl << std::endl;
	std::cout << "tmp = " << std::endl << mat.minIndex() << std::endl << std::endl;

	Cumat::Matrixd mat3(2000, 2000);
	Cumat::Matrixd tmp2(2000, 2001);
	Cumat::Matrixd tmp3(2000, 2001);

	Cumat::Matrixd Wy(Cumat::Matrixd::random(512, 512));
	Cumat::Matrixd dy(512, 512);
	Cumat::Matrixd tmp4(Wy.rows(), dy.cols());

	for (int i = 0; i < 10; i++) {
		Wy.mmul(dy, tmp4) += 1;
		Wy.swap(tmp4);
	}
	// std::cout << typeid(tmp + tmp).name() << std::endl;

	tmp.fill(0.2);
	tmp5.fill(0.02);
	tmp = tmp + tmp5 + tmp5 + tmp;
	tmp = tmp5 + tmp;
	// tmp = tmp + tmp5 + tmp5 + tmp;
	// tmp = tmp + tmp;
	tmp = tmp + tmp + tmp + tmp + tmp + tmp + tmp;
	auto start = std::chrono::high_resolution_clock::now();
	tmp = tmp + tmp + tmp + tmp + tmp + tmp + tmp;
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	start = std::chrono::high_resolution_clock::now();
	tmp = tmp + tmp5 + tmp + tmp5;
	end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
	// std::cout << tmp << std::endl;

	// std::cout << Wy << std::endl;

	Cumat::destroyCublasHandle();

	return 0;
}
