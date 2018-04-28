#include <iostream>
#include <chrono>
#include "Core"

int main(int argc, char const* argv[])
{
	Cumat::init();
	Cumat::Matrixf mat(std::move(Cumat::Matrixf::random(5, 5)));
	Cumat::Matrixf mat2(std::move(Cumat::Matrixf::random(5, 5)));
	Cumat::Matrixd dmat(5, 5, 1);
	Cumat::Matrixf result(5, 5);

	Cumat::Matrixf vec1(66, 1, 1);
	Cumat::Matrixf vec2(66, 1, 1);
	Cumat::Matrixf vec3(66, 1, 1);
	Cumat::Matrixf vec4(66, 1, 1);

	Cumat::CudaHandler::setStream(1);
	vec1 = vec1 + vec2;
	Cumat::CudaHandler::setStream(2);
	vec3 = vec2 + vec2;
	Cumat::CudaHandler::setStream(3);
	vec4 = vec2 + vec2;
	Cumat::CudaHandler::setDefaultStream();
	cudaDeviceSynchronize();

	result.identity();

	std::cout << (~(~pow(~mat, mat) + mat)).eval() << std::endl << std::endl;
	std::cout << mmul(mat, result / 2.0f).eval<float>() << std::endl << std::endl;
	cudaDeviceSynchronize();

	(mat2 + mat2).eval<float>();

	std::cout << "A = " << std::endl << mat << std::endl << std::endl;
	std::cout << "B = " << std::endl << mat2 << std::endl << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	mat = mmul(mat + mat, result + result);
	// mat = mmul(mat, mat2);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Result = " << std::endl << mat << std::endl << std::endl;

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::Matrixf large1(std::move(Cumat::Matrixf::random(4096, 4096)));
	Cumat::Matrixf large2(std::move(Cumat::Matrixf::random(4096, 4096)));
	Cumat::Matrixd large3(std::move(Cumat::Matrixd::random(4096, 4096)));

	// large1.mmul(large2, large2);
	large1.transpose();
	cudaDeviceSynchronize();
	start = std::chrono::high_resolution_clock::now();
	large1.transpose(large2);
	// large1.mmul(large2, large2);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	std::cout << large1(12, 13) << " " << large3(12, 13) << std::endl;
	large1(12, 13) += large3(12, 13);
	std::cout << large1(12, 13) << " " << large1(456, 132) << std::endl;
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	Cumat::end();

	return 0;
}
