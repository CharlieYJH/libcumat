#include <iostream>
#include <chrono>
#include <typeinfo>
#include "Core"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixf mat(std::move(Cumat::Matrixf::random(5, 5)));
	Cumat::Matrixf mat2(std::move(Cumat::Matrixf::random(5, 5)));
	Cumat::Matrixd dmat(5, 5, 1);
	Cumat::Matrixf result(5, 5);

	for (int i = 0; i < result.rows(); i++)
		for (int j = 0; j < result.cols(); j++)
			if (i == j)
				result.set(i, j, 1);

	std::cout << (~(~pow(~mat, mat) + mat)).eval() << std::endl << std::endl;
	std::cout << mmul(mat, result / 2.0f).eval<float>() << std::endl << std::endl;
	cudaDeviceSynchronize();

	(mat2 + mat2).eval<float>();

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	mat = mmul(mat + mat, result + result);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Result = " << std::endl << mat << std::endl << std::endl;
	mat /= 2;
	std::cout << "Result = " << std::endl << mat << std::endl << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::Matrixf large1(std::move(Cumat::Matrixf::random(4096, 4096)));
	Cumat::Matrixf large2(std::move(Cumat::Matrixf::random(4096, 4096)));
	Cumat::Matrixd large3(std::move(Cumat::Matrixd::random(4096, 4096)));

	large1 = mmul(large2, large2);
	large1 /= large3;
	cudaDeviceSynchronize();
	start = std::chrono::high_resolution_clock::now();
	// large1 = mmul(large2, large2);
	large3.rand();
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
