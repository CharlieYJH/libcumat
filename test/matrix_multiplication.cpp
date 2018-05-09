#include "catch.hpp"
#include "GPUCompare.hpp"
#include "CPUMatrix.hpp"
#include "Core"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix multiplication", "[matrix][multiplication][float]")
{
	Cumat::init();

	srand(time(0));

	size_t m = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t n = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t k = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;

	CPUMatrix<float> cpu_mat1(m, n);
	cpu_mat1.rand();

	CPUMatrix<float> cpu_mat2(n, k);
	cpu_mat2.rand();

	CPUMatrix<float> cpu_mat3(m, k);

	Cumat::Matrixf gpu_mat1(cpu_mat1.data);
	gpu_mat1.resize(m, n);
	REQUIRE(gpu_mat1.rows() == m);
	REQUIRE(gpu_mat1.cols() == n);
	REQUIRE(gpu_mat1.size() == m * n);
	REQUIRE_FALSE(approxEqual(gpu_mat1, 0));

	for (size_t i = 0; i < gpu_mat1.size(); ++i)
		REQUIRE(gpu_mat1(i) == Approx(cpu_mat1.data[i]));

	Cumat::Matrixf gpu_mat2(cpu_mat2.data);
	gpu_mat2.resize(n, k);
	REQUIRE(gpu_mat2.rows() == n);
	REQUIRE(gpu_mat2.cols() == k);
	REQUIRE(gpu_mat2.size() == n * k);
	REQUIRE_FALSE(approxEqual(gpu_mat2, 0));

	for (size_t i = 0; i < gpu_mat2.size(); ++i)
		REQUIRE(gpu_mat2(i) == Approx(cpu_mat2.data[i]));

	Cumat::Matrixf gpu_mat3;
	REQUIRE(gpu_mat3.rows() == 0);
	REQUIRE(gpu_mat3.cols() == 0);
	REQUIRE(gpu_mat3.size() == 0);

	SECTION("Matrix class matrix multiplication method (Zero beta)")
	{
		gpu_mat3.mmul(gpu_mat1, gpu_mat2);
		CPUMatrixMultiply(cpu_mat1, cpu_mat2, cpu_mat3);

		REQUIRE(gpu_mat3.rows() == m);
		REQUIRE(gpu_mat3.cols() == k);
		REQUIRE(gpu_mat3.size() == m * k);

		for (size_t i = 0; i < gpu_mat3.size(); ++i)
			CHECK(gpu_mat3(i) == Approx(cpu_mat3.data[i]).epsilon(0.01));
	}

	SECTION("Matrix class matrix multiplication method (Non-zero beta)")
	{
		gpu_mat3.mmul(gpu_mat1, gpu_mat2);
		CPUMatrixMultiply(cpu_mat1, cpu_mat2, cpu_mat3);

		CPUMatrix<float> cpu_mat4 = cpu_mat3;
		REQUIRE(cpu_mat4.data == cpu_mat3.data);

		REQUIRE(gpu_mat3.rows() == m);
		REQUIRE(gpu_mat3.cols() == k);
		REQUIRE(gpu_mat3.size() == m * k);

		for (size_t i = 0; i < gpu_mat3.size(); ++i)
			CHECK(gpu_mat3(i) == Approx(cpu_mat3.data[i]).epsilon(0.01));

		gpu_mat3.mmul(gpu_mat1, gpu_mat2, 2.3);

		for (size_t i = 0; i < cpu_mat4.data.size(); ++i)
			cpu_mat4.data[i] += 2.3 * cpu_mat3.data[i];

		REQUIRE(gpu_mat3.rows() == m);
		REQUIRE(gpu_mat3.cols() == k);
		REQUIRE(gpu_mat3.size() == m * k);

		for (size_t i = 0; i < gpu_mat3.size(); ++i)
			CHECK(gpu_mat3(i) == Approx(cpu_mat4.data[i]).epsilon(0.01));
	}

	SECTION("Matrix multiply function")
	{
		gpu_mat3 = mmul(gpu_mat1, gpu_mat2);
		CPUMatrixMultiply(cpu_mat1, cpu_mat2, cpu_mat3);

		REQUIRE(gpu_mat3.rows() == m);
		REQUIRE(gpu_mat3.cols() == k);
		REQUIRE(gpu_mat3.size() == m * k);

		for (size_t i = 0; i < gpu_mat3.size(); ++i)
			CHECK(gpu_mat3(i) == Approx(cpu_mat3.data[i]).epsilon(0.01));
	}

	SECTION("Matrix multiply function (with expressions)")
	{
		gpu_mat3 = mmul(1.3 * exp(gpu_mat1) + 2, tanh(gpu_mat2) * 2 - 3.4f);

		for (size_t i = 0; i < cpu_mat1.data.size(); ++i)
			cpu_mat1.data[i] = 1.3 * std::exp(cpu_mat1.data[i]) + 2;

		for (size_t i = 0; i < cpu_mat2.data.size(); ++i)
			cpu_mat2.data[i] = std::tanh(cpu_mat2.data[i]) * 2 - 3.4f;

		CPUMatrixMultiply(cpu_mat1, cpu_mat2, cpu_mat3);

		REQUIRE(gpu_mat3.rows() == m);
		REQUIRE(gpu_mat3.cols() == k);
		REQUIRE(gpu_mat3.size() == m * k);

		for (size_t i = 0; i < gpu_mat3.size(); ++i)
			CHECK(gpu_mat3(i) == Approx(cpu_mat3.data[i]).epsilon(0.01));
	}

	Cumat::end();
}
