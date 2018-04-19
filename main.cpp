#include <iostream>
#include <chrono>
#include <typeinfo>
#include "inc/libcumat.h"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixf mat(std::move(Cumat::Matrixf::random(5, 6)));
	Cumat::Matrixf mat2(std::move(Cumat::Matrixf::random(5, 6)));
	Cumat::Matrixf result(6, 6);

	for (int i = 0; i < result.rows(); i++)
		for (int j = 0; j < result.cols(); j++)
			if (i == j)
				result.set(i, j, 1);

	std::cout << mat << std::endl << std::endl;
	std::cout << mmuld(mat, result / 2.0).eval<float>() << std::endl << std::endl;
	cudaDeviceSynchronize();

	(mat2 + mat2).eval();

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	mat = mmuld(mat + mat, result + result);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Result = " << std::endl << mat << std::endl << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
