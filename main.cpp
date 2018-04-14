#include <iostream>
#include <chrono>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixd mat(std::move(Cumat::Matrixd::random(1024, 1024)));
	Cumat::Matrixd mat2(std::move(Cumat::Matrixd::random(1024, 1024)));
	Cumat::Matrixd result(1024, 1024);
	// Cumat::Matrixd result2(2000, 2000);
	// Cumat::Matrixd tmp(5, 6);
	// Cumat::Matrixd tmp5(5, 6);

	// std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	// std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	// (mat + mat).eval();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1000; i++) {
		result.mmul(mat, mat2, 0);
	}
	cudaDeviceSynchronize();
	// result2.mmul(~mat2, ~mat, 1);
	auto end = std::chrono::high_resolution_clock::now();

	// std::cout << "A x B = " << std::endl << result << std::endl << std::endl;
	// std::cout << "tmp = " << std::endl << mat.minElement() << std::endl << std::endl;
	// std::cout << "tmp = " << std::endl << mat.minIndex() << std::endl << std::endl;

	// Cumat::Matrixd mat3(2000, 2001);
	// Cumat::Matrixd tmp2(2000, 2001);
	// Cumat::Matrixd tmp3(2000, 2001);

	// Cumat::Matrixd Wy(Cumat::Matrixd::random(512, 512));
	// Cumat::Matrixd dy(512, 512);
	// Cumat::Matrixd tmp4(Wy.rows(), dy.cols());

	// for (int i = 0; i < 10; i++) {
		// // Wy.mmul(dy, tmp4) += 1;
		// // Wy.swap(tmp4);
	// }

	// tmp.fill(0.2);
	// tmp5.fill(0.02);

	// tmp = 1.0 + tmp5 + 1.0;
	// Cumat::Matrixf fmat(6, 5);
	// fmat.fill(-1);

	// tmp.rand();
	// tmp5.rand();

	// // std::cout << (~exp2(~abs(-2 * 3 + -(fmat + fmat)) / (2 - 2 / tmp5 - (1 + 1)) - (tmp / 3) * (-2.0 * tmp))).eval() << std::endl;

	// // std::cout << tmp << std::endl << std::endl;
	// tmp.exp().log().exp();
	// // mat3.transpose();
	// // tmp = -1.0 + 1.0 + 4.3 + tmp + 2.3;
	// // std::cout << tmp << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
