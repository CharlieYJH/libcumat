#include <iostream>
#include <chrono>
#include <typeinfo>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(5, 3)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(3, 6)));
	Cumat::Matrixd tmp(5, 6);
	Cumat::Matrixd tmp5(5, 6);

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

	tmp.fill(0.2);
	tmp5.fill(0.02);

	tmp = 1.0 + tmp5 + 1.0;
	Cumat::Matrixf fmat(6, 5);
	fmat.fill(-1);

	std::cout << (exp2(~abs(-2 * 3 + -(fmat + fmat)) / (2 - 2 / tmp5 - (1 + 1)) - (tmp / 3) * (-2.0 * tmp * tmp))).eval() << std::endl;

	// std::cout << tmp << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	mat3 = mat3.transpose();
	// tmp = -1.0 + 1.0 + 4.3 + tmp + 2.3;
	auto end = std::chrono::high_resolution_clock::now();
	// std::cout << tmp << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
