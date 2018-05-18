#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix number fill", "[fill][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	REQUIRE(mat1_size > 1);

	Cumat::Matrixf mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, 0.0f));

	SECTION("Float fill")
	{
		float val = 0.025f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = 10.045f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = -5604.12f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = -0.000450123f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));
	}

	SECTION("Integer fill")
	{
		int val = 1;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = 2031;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = -313;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = -13023;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));
	}

	SECTION("Double fill")
	{
		double val = 0.00012310;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = 123415.231;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = -23.1923818;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));

		val = -0.013828101;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (float)val));
	}
}

TEST_CASE("Double matrix number fill", "[fill][double]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	REQUIRE(mat1_size > 1);

	Cumat::Matrixd mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, 0.0));

	SECTION("Float fill")
	{
		float val = 0.023193f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = 8.1382f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = -68182.12f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = -0.000450123f;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));
	}

	SECTION("Integer fill")
	{
		int val = 19;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = 203191;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = -913929;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));

		val = -13;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, (double)val));
	}

	SECTION("Double fill")
	{
		double val = 0.000019231;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = 5881382.1823;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = -19923.182318;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));

		val = -0.0001291901823;

		mat1.fill(val);
		REQUIRE(approxEqual(mat1, val));
	}
}

TEST_CASE("Float matrix zero fill", "[zero][fill][float]")
{
	Cumat::init();
	
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	REQUIRE(mat1_size > 1);

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0f));

	mat1.zero();
	REQUIRE(approxEqual(mat1, 0.0f));

	Cumat::end();
}

TEST_CASE("Double matrix zero fill", "[zero][fill][double]")
{
	Cumat::init();
	
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	REQUIRE(mat1_size > 1);

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	mat1.zero();
	REQUIRE(approxEqual(mat1, 0.0));

	Cumat::end();
}
