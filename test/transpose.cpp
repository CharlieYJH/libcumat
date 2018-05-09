#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "Core"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix transpose", "[transpose][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	SECTION("In-place transpose")
	{
		Cumat::Matrixf mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		mat2.transpose();

		REQUIRE(mat2.rows() == mat1_cols);
		REQUIRE(mat2.cols() == mat1_rows);
		REQUIRE(mat2.size() == mat1_size);
		REQUIRE(approxEqual(mat2, (~mat1).eval()));
		REQUIRE(approxEqual(mat2, transpose(mat1).eval()));
	}

	SECTION("Out-of-place transpose")
	{
		Cumat::Matrixf mat2;
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);

		Cumat::Matrixf mat3 = mat1;
		REQUIRE(approxEqual(mat1, mat3));

		mat2.transpose(mat1);

		REQUIRE(mat2.rows() == mat1_cols);
		REQUIRE(mat2.cols() == mat1_rows);
		REQUIRE(mat2.size() == mat1_size);
		REQUIRE(approxEqual(mat2, (~mat1).eval()));
		REQUIRE(approxEqual(mat2, transpose(mat1).eval()));
		REQUIRE(approxEqual(mat1, mat3));
	}

	Cumat::end();
}

TEST_CASE("Double matrix transpose", "[transpose][double]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	SECTION("In-place transpose")
	{
		Cumat::Matrixd mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		mat2.transpose();

		REQUIRE(mat2.rows() == mat1_cols);
		REQUIRE(mat2.cols() == mat1_rows);
		REQUIRE(mat2.size() == mat1_size);
		REQUIRE(approxEqual(mat2, (~mat1).eval<double>()));
		REQUIRE(approxEqual(mat2, transpose(mat1).eval<double>()));
	}

	SECTION("Out-of-place transpose")
	{
		Cumat::Matrixd mat2;
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);

		Cumat::Matrixd mat3 = mat1;
		REQUIRE(approxEqual(mat1, mat3));

		mat2.transpose(mat1);

		REQUIRE(mat2.rows() == mat1_cols);
		REQUIRE(mat2.cols() == mat1_rows);
		REQUIRE(mat2.size() == mat1_size);
		REQUIRE(approxEqual(mat2, (~mat1).eval<double>()));
		REQUIRE(approxEqual(mat2, transpose(mat1).eval<double>()));
		REQUIRE(approxEqual(mat1, mat3));
	}

	Cumat::end();
}

TEST_CASE("Float column matrix transpose", "[thin][column][transpose][float]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = 1;
	size_t mat1_size = mat1_rows;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(approxEqual(mat1, mat2));

	mat2.transpose();

	REQUIRE(mat2.rows() == mat1_cols);
	REQUIRE(mat2.cols() == mat1_rows);
	REQUIRE(mat2.size() == mat1_size);
	REQUIRE(approxEqual(mat2, (~mat1).eval()));
	REQUIRE(approxEqual(mat2, transpose(mat1).eval()));

	Cumat::end();
}

TEST_CASE("Double column matrix transpose", "[thin][column][transpose][double]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = 1;
	size_t mat1_size = mat1_rows;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixd mat2 = mat1;
	REQUIRE(approxEqual(mat1, mat2));

	mat2.transpose();

	REQUIRE(mat2.rows() == mat1_cols);
	REQUIRE(mat2.cols() == mat1_rows);
	REQUIRE(mat2.size() == mat1_size);
	REQUIRE(approxEqual(mat2, (~mat1).eval<double>()));
	REQUIRE(approxEqual(mat2, transpose(mat1).eval<double>()));

	Cumat::end();
}

TEST_CASE("Float row matrix transpose", "[thin][row][transpose][float]")
{
	Cumat::init();

	size_t mat1_rows = 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(approxEqual(mat1, mat2));

	mat2.transpose();

	REQUIRE(mat2.rows() == mat1_cols);
	REQUIRE(mat2.cols() == mat1_rows);
	REQUIRE(mat2.size() == mat1_size);
	REQUIRE(approxEqual(mat2, (~mat1).eval()));
	REQUIRE(approxEqual(mat2, transpose(mat1).eval()));

	Cumat::end();
}

TEST_CASE("Double row matrix transpose", "[thin][row][transpose][double]")
{
	Cumat::init();

	size_t mat1_rows = 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixd mat2 = mat1;
	REQUIRE(approxEqual(mat1, mat2));

	mat2.transpose();

	REQUIRE(mat2.rows() == mat1_cols);
	REQUIRE(mat2.cols() == mat1_rows);
	REQUIRE(mat2.size() == mat1_size);
	REQUIRE(approxEqual(mat2, (~mat1).eval<double>()));
	REQUIRE(approxEqual(mat2, transpose(mat1).eval<double>()));

	Cumat::end();
}
