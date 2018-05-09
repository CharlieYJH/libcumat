#include "catch.hpp"
#include "GPUCompare.hpp"
#include "Core"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix assignments", "[assignment][float]")
{
	Cumat::init();

	srand(time(0));

	size_t rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t size = rows * cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(rows, cols);
	REQUIRE(mat1.rows() == rows);
	REQUIRE(mat1.cols() == cols);
	REQUIRE(mat1.size() == size);

	Cumat::Matrixf mat2 = Cumat::Matrixf::random(rows, cols);
	REQUIRE(mat2.rows() == rows);
	REQUIRE(mat2.cols() == cols);
	REQUIRE(mat2.size() == size);

	Cumat::Matrixf mat3 = Cumat::Matrixf::random(rows, cols);
	REQUIRE(mat3.rows() == rows);
	REQUIRE(mat3.cols() == cols);
	REQUIRE(mat3.size() == size);

	Cumat::Matrixf mat4 = mat1;
	REQUIRE(approxEqual(mat4, mat1));

	SECTION("Regular assignment")
	{
		mat4 = mat2;
		REQUIRE_FALSE(approxEqual(mat4, mat1));
		REQUIRE_FALSE(approxEqual(mat4, mat3));
		REQUIRE(approxEqual(mat4, mat2));

		mat3 = mat1;
		REQUIRE_FALSE(approxEqual(mat3, mat4));
		REQUIRE_FALSE(approxEqual(mat3, mat2));
		REQUIRE(approxEqual(mat3, mat1));

		mat4 = mat3;
		REQUIRE_FALSE(approxEqual(mat4, mat2));
		REQUIRE(approxEqual(mat4, mat3));
		REQUIRE(approxEqual(mat4, mat1));
	}

	SECTION("Addtion assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() + mat3(i).val() * 3.4 + std::exp(mat2(i).val() / 2.4);

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 += mat3 * 3.4 + exp(mat2 / 2.4);

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Subtraction assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() - (mat2(i).val() - mat1(i).val() + mat3(i).val() * 0.4 * mat3(i).val());

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 -= mat2 - mat1 + mat3 * 0.4 * mat3;

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Multiplication assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() * (mat2(i).val() + mat3(i).val() * mat3(i).val() * mat3(i).val() * 0.34f + mat1(i).val());

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 *= mat2 + cube(mat3) * 0.34f + mat1;

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Division assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() / (std::abs(mat3(i).val()) * 100.323 + std::cbrt(mat1(i).val()));

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 /= abs(mat3) * 100.323 + cbrt(mat1);

		REQUIRE(approxEqual(mat1, mat4, 1e-3f));
	}

	Cumat::end();
}

TEST_CASE("Double matrix assignments", "[assignment][double]")
{
	Cumat::init();

	size_t rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t size = rows * cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(rows, cols);
	REQUIRE(mat1.rows() == rows);
	REQUIRE(mat1.cols() == cols);
	REQUIRE(mat1.size() == size);

	Cumat::Matrixd mat2 = Cumat::Matrixd::random(rows, cols);
	REQUIRE(mat2.rows() == rows);
	REQUIRE(mat2.cols() == cols);
	REQUIRE(mat2.size() == size);

	Cumat::Matrixd mat3 = Cumat::Matrixd::random(rows, cols);
	REQUIRE(mat3.rows() == rows);
	REQUIRE(mat3.cols() == cols);
	REQUIRE(mat3.size() == size);

	Cumat::Matrixd mat4 = mat1;
	REQUIRE(approxEqual(mat4, mat1));

	SECTION("Regular assignment")
	{
		mat4 = mat2;
		REQUIRE_FALSE(approxEqual(mat4, mat1));
		REQUIRE_FALSE(approxEqual(mat4, mat3));
		REQUIRE(approxEqual(mat4, mat2));

		mat3 = mat1;
		REQUIRE_FALSE(approxEqual(mat3, mat4));
		REQUIRE_FALSE(approxEqual(mat3, mat2));
		REQUIRE(approxEqual(mat3, mat1));

		mat4 = mat3;
		REQUIRE_FALSE(approxEqual(mat4, mat2));
		REQUIRE(approxEqual(mat4, mat3));
		REQUIRE(approxEqual(mat4, mat1));
	}

	SECTION("Addtion assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() + mat3(i).val() * 3.4 + std::exp(mat2(i).val() / 2.4);

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 += mat3 * 3.4 + exp(mat2 / 2.4);

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Subtraction assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() - (mat2(i).val() - mat1(i).val() + mat3(i).val() * 0.4 * mat3(i).val());

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 -= mat2 - mat1 + mat3 * 0.4 * mat3;

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Multiplication assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() * (mat2(i).val() + mat3(i).val() * mat3(i).val() * mat3(i).val() * 0.34f + mat1(i).val());

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 *= mat2 + cube(mat3) * 0.34f + mat1;

		REQUIRE(approxEqual(mat1, mat4));
	}

	SECTION("Division assignment")
	{
		for (size_t i = 0; i < mat4.size(); ++i)
			mat4(i) = mat1(i).val() / (std::abs(mat3(i).val()) * 100.323 + std::cbrt(mat1(i).val()));

		REQUIRE_FALSE(approxEqual(mat1, mat4));

		mat1 /= abs(mat3) * 100.323 + cbrt(mat1);

		REQUIRE(approxEqual(mat1, mat4));
	}

	Cumat::end();
}
