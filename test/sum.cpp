#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix sum", "[sum][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0f));

	float sum = 0.0f;

	for (size_t i = 0; i < mat1.size(); ++i)
		sum += mat1(i).val();

	REQUIRE(mat1.sum() == Approx(sum).epsilon(0.05));

	Cumat::end();
}

TEST_CASE("Double matrix sum", "[sum][double]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	double sum = 0.0;

	for (size_t i = 0; i < mat1.size(); ++i)
		sum += mat1(i).val();

	REQUIRE(mat1.sum() == Approx(sum));

	Cumat::end();
}
